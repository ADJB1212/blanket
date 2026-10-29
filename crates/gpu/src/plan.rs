//! Reusable, same-size unary pipelines and bounded tile geometry.
use crate::Descriptor;
use crate::protocol::operation;

#[derive(Clone)]
pub struct Step {
    pub operation: u32,
    pub kernel: u32,
    pub axis: u32,
    pub values: Vec<f32>,
}

pub struct Plan {
    descriptor: Descriptor,
    steps: Vec<Step>,
    tile: [u32; 2],
    halo: [u32; 2],
    scratch_bytes: usize,
}

impl Plan {
    pub fn new(descriptor: Descriptor, steps: Vec<Step>, memory_limit: usize) -> Result<Self, String> {
        descriptor.byte_len()?;
        if steps.len() > 1024 {
            return Err("GPU plans support at most 1024 steps".into());
        }
        let mut halo = [0u32; 2];
        let mut fused: Vec<Step> = Vec::new();
        for step in steps {
            let k = step.kernel as usize;
            let valid = match step.operation {
                operation::COPY | operation::INVERT => step.values.is_empty(),
                operation::BRIGHTNESS => step.values.len() == 1 && step.values[0].is_finite(),
                operation::QUANTIZE => descriptor.float && step.values.len() == 1 && matches!(step.values[0], 1023.0 | 4095.0 | 65535.0),
                operation::POINT => {
                    !descriptor.float
                        && step.values.len() == 256 * descriptor.channels as usize
                        && step.values.iter().all(|v| v.is_finite() && v.fract() == 0.0 && (0.0..=255.0).contains(v))
                }
                operation::CONVOLVE => matches!(k, 3 | 5) && step.values.len() == k * k + 2 && step.values.iter().all(|v| v.is_finite()),
                operation::BOX_BLUR => {
                    step.axis <= 1 && k <= i32::MAX as usize / 2 && step.values.len() == 2 && step.values.iter().all(|v| v.is_finite() && *v >= 0.0)
                }
                operation::RANK => matches!(k, 1 | 3 | 5 | 7) && (step.axis as usize) < k * k && step.values.is_empty(),
                _ => false,
            };
            if !valid {
                return Err("unsupported or invalid GPU plan step".into());
            }
            let radius = match step.operation {
                operation::CONVOLVE | operation::RANK => [step.kernel / 2; 2],
                operation::BOX_BLUR => {
                    let mut radius = [0; 2];
                    radius[step.axis as usize] = step.kernel + 1;
                    radius
                }
                _ => [0; 2],
            };
            for axis in 0..2 {
                halo[axis] = halo[axis].checked_add(radius[axis]).ok_or("GPU plan halo overflow")?;
            }
            if step.operation == operation::COPY {
                continue;
            }
            if !descriptor.float && matches!(step.operation, operation::INVERT | operation::POINT) {
                let table = if step.operation == operation::INVERT {
                    (0..256 * descriptor.channels).map(|v| (255 - v % 256) as f32).collect()
                } else {
                    step.values
                };
                if let Some(previous) = fused.last_mut().filter(|step| step.operation == operation::POINT) {
                    for (i, value) in previous.values.iter_mut().enumerate() {
                        *value = table[i / 256 * 256 + *value as usize];
                    }
                } else {
                    fused.push(Step { operation: operation::POINT, kernel: 0, axis: 0, values: table });
                }
            } else {
                fused.push(step);
            }
        }
        let bytes_per_pixel = descriptor.channels as usize * if descriptor.float { 4 } else { 1 };
        let fits = |width: u32, height: u32| -> Option<usize> {
            let width = u64::from(width).saturating_add(2 * u64::from(halo[0])).min(u64::from(descriptor.width));
            let height = u64::from(height).saturating_add(2 * u64::from(halo[1])).min(u64::from(descriptor.height));
            usize::try_from(width * height).ok()?.checked_mul(bytes_per_pixel)?.checked_mul(2)
        };
        let mut tile = [descriptor.width, descriptor.height];
        while fits(tile[0], tile[1]).is_none_or(|bytes| bytes > memory_limit) {
            let axis = usize::from(tile[1] > tile[0]);
            if tile[axis] == 1 {
                return Err("GPU working-memory limit cannot hold one tile and its filter halos".into());
            }
            tile[axis] = tile[axis].div_ceil(2);
        }
        let scratch_bytes = fits(tile[0], tile[1]).ok_or("GPU scratch size overflow")?;
        Ok(Self { descriptor, steps: fused, tile, halo, scratch_bytes })
    }

    pub fn descriptor(&self) -> Descriptor {
        self.descriptor
    }
    pub fn steps(&self) -> &[Step] {
        &self.steps
    }
    pub fn tile(&self) -> [u32; 2] {
        self.tile
    }
    pub fn halo(&self) -> [u32; 2] {
        self.halo
    }
    pub fn scratch_bytes(&self) -> usize {
        self.scratch_bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn descriptor(float: bool) -> Descriptor {
        Descriptor { width: 101, height: 79, channels: 4, float }
    }
    fn step(operation: u32) -> Step {
        Step { operation, kernel: 0, axis: 0, values: Vec::new() }
    }
    #[test]
    fn point_fusion_preserves_every_channel_value() {
        let values: Vec<f32> = (0..1024).map(|v| (v % 17) as f32).collect();
        let mut lookup = step(operation::POINT);
        lookup.values = values.clone();
        let plan = Plan::new(descriptor(false), vec![step(operation::INVERT), lookup, step(operation::INVERT)], 100_000).unwrap();
        assert_eq!(plan.steps.len(), 1);
        for band in 0..4 {
            for value in 0..256 {
                assert_eq!(plan.steps[0].values[band * 256 + value], 255.0 - values[band * 256 + 255 - value]);
            }
        }
        let plan = Plan::new(descriptor(true), vec![step(operation::INVERT), step(operation::INVERT)], 1_000_000).unwrap();
        assert_eq!(plan.steps.len(), 2);
    }
    #[test]
    fn tiles_account_for_cumulative_halos_and_sample_storage() {
        let rank = Step { operation: operation::RANK, kernel: 7, axis: 24, values: Vec::new() };
        let blur = Step { operation: operation::BOX_BLUR, kernel: 2, axis: 0, values: vec![0.2, 0.1] };
        let plan = Plan::new(descriptor(true), vec![rank, blur], 16_384).unwrap();
        assert_eq!(plan.halo, [6, 3]);
        assert!(plan.scratch_bytes <= 16_384);
        assert!(plan.tile[0] < 101 || plan.tile[1] < 79);
        assert!(Plan::new(descriptor(true), vec![], 1).is_err());
    }
    #[test]
    fn rejects_unsupported_steps_and_invalid_lookup_indices() {
        assert!(Plan::new(descriptor(false), vec![step(operation::RESIZE)], 100_000).is_err());
        let lookup = Step { values: vec![256.0; 1024], ..step(operation::POINT) };
        assert!(Plan::new(descriptor(false), vec![lookup], 100_000).is_err());
    }
}
