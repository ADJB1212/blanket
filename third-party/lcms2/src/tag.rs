use std::{mem, ptr};

use foreign_types::ForeignTypeRef;

use crate::*;

unsafe fn cast<'a, T>(ptr: *const u8) -> Option<&'a T> {
    if 0 == ptr.align_offset(align_of::<T>()) { unsafe { ptr.cast::<T>().as_ref() } } else { None }
}

fn byte_ptr<T>(ptr: &T) -> *const u8 {
    ptr::from_ref(ptr).cast()
}

/// `from_ptr()` methods require mut for no good reason
unsafe fn aligned_mut<T>(ptr: *const u8) -> *mut T {
    assert!(0 == ptr.align_offset(align_of::<T>()), "Tag data pointer must be aligned");
    ptr.cast_mut().cast::<T>()
}

unsafe fn read_u32_enum_tag<T: TryFrom<u32>>(data: *const u8) -> Option<T> {
    assert_eq!(4, size_of::<T>());
    T::try_from(unsafe { data.cast::<u32>().read_unaligned() }).ok()
}

macro_rules! check_enum_field {
    ($val:expr, $struct_ty:ty, $field:ident, $enum_ty:ty) => {{
        let struct_ptr: *const $struct_ty = $val;
        let field_ptr = struct_ptr.cast::<u8>().add(mem::offset_of!($struct_ty, $field));
        read_u32_enum_tag::<$enum_ty>(field_ptr).is_some()
    }};
}

impl Tag<'_> {
    #[must_use]
    pub fn is_none(&self) -> bool {
        matches!(*self, Tag::None)
    }

    #[must_use]
    pub unsafe fn data_for_signature(&self, sig: TagSignature) -> *const u8 {
        unsafe { self.data_for_signature_opt(sig).unwrap_or_else(|| panic!("Signature type {sig:?} does not support this tag data type")) }
    }

    #[must_use]
    pub unsafe fn data_for_signature_opt(&self, sig: TagSignature) -> Option<*const u8> {
        use crate::TagSignature::*;
        Some(match (sig, self) {
            (RedColorantTag | BlueColorantTag | GreenColorantTag | LuminanceTag | MediaBlackPointTag | MediaWhitePointTag, &Tag::CIEXYZ(data)) => {
                byte_ptr::<ffi::CIEXYZ>(data)
            }
            (
                ViewingCondDescTag
                | CharTargetTag
                | CopyrightTag
                | DeviceMfgDescTag
                | DeviceModelDescTag
                | ProfileDescriptionTag
                | ProfileDescriptionMLTag
                | ScreeningDescTag
                | CrdInfoTag,
                &Tag::MLU(data),
            ) => data.as_ptr().cast(),
            (ChromaticityTag | ChromaticAdaptationTag, &Tag::CIExyYTRIPLE(data)) => byte_ptr::<ffi::CIExyYTRIPLE>(data),
            (ColorantTableTag | ColorantTableOutTag | NamedColor2Tag, &Tag::NamedColorList(data)) => data.as_ptr().cast(),
            (DataTag | Ps2CRD0Tag | Ps2CRD1Tag | Ps2CRD2Tag | Ps2CRD3Tag | Ps2CSATag, &Tag::ICCData(data)) => {
                ptr::from_ref::<ffi::ICCData>(data).cast()
            }
            (Ps2RenderingIntentTag, &Tag::ICCData(data)) => ptr::from_ref::<ffi::ICCData>(data).cast(),
            (
                AToB0Tag | AToB1Tag | AToB2Tag | BToA0Tag | BToA1Tag | BToA2Tag | DToB0Tag | DToB1Tag | DToB2Tag | DToB3Tag | BToD0Tag | BToD1Tag
                | BToD2Tag | BToD3Tag | GamutTag | Preview0Tag | Preview1Tag | Preview2Tag,
                &Tag::Pipeline(data),
            ) => data.as_ptr().cast(),
            (BlueTRCTag | GrayTRCTag | GreenTRCTag | RedTRCTag, &Tag::ToneCurve(data)) => data.as_ptr().cast(),
            (ColorimetricIntentImageStateTag, Tag::ColorimetricIntentImageState(owned)) => byte_ptr::<ffi::ColorimetricIntentImageState>(owned),
            (PerceptualRenderingIntentGamutTag | SaturationRenderingIntentGamutTag, Tag::Intent(owned)) => byte_ptr::<ffi::Intent>(owned),
            (TechnologyTag, Tag::Technology(data)) => byte_ptr::<ffi::TechnologySignature>(data),
            (MeasurementTag, &Tag::ICCMeasurementConditions(data)) => byte_ptr::<ffi::ICCMeasurementConditions>(data),
            (ProfileSequenceDescTag | ProfileSequenceIdTag, &Tag::SEQ(data)) => byte_ptr::<ffi::SEQ>(data),
            (ScreeningTag, &Tag::Screening(data)) => byte_ptr::<ffi::Screening>(data),
            (UcrBgTag, &Tag::UcrBg(data)) => byte_ptr::<ffi::UcrBg>(data),
            (VcgtTag, Tag::VcgtCurves(arr)) => ptr::from_ref::<[_; 3]>(arr).cast(),
            (ViewingConditionsTag, &Tag::ICCViewingConditions(data)) => byte_ptr::<ffi::ICCViewingConditions>(data),
            (CicpTag, &Tag::VideoSignal(data)) => byte_ptr::<ffi::VideoSignalType>(data),
            (MHC2Tag, &Tag::MHC2(data)) => byte_ptr::<ffi::MHC2Type>(data),
            (_sig, _) => return None,
        })
    }

    #[must_use]
    pub unsafe fn new(sig: TagSignature, data: *const u8) -> Self {
        unsafe {
            if data.is_null() {
                return Tag::None;
            }
            use crate::TagSignature::*;
            match sig {
                BlueColorantTag | GreenColorantTag | LuminanceTag | MediaBlackPointTag | MediaWhitePointTag | RedColorantTag => {
                    cast(data).map(Tag::CIEXYZ).unwrap_or(Tag::None)
                }
                CharTargetTag
                | CopyrightTag
                | CrdInfoTag
                | DeviceMfgDescTag
                | DeviceModelDescTag
                | ProfileDescriptionTag
                | ProfileDescriptionMLTag
                | ScreeningDescTag
                | ViewingCondDescTag => Tag::MLU(MLURef::from_ptr(aligned_mut(data))),
                ChromaticityTag | ChromaticAdaptationTag => cast(data).map(Tag::CIExyYTRIPLE).unwrap_or(Tag::None),
                ColorantTableTag | ColorantTableOutTag | NamedColor2Tag => Tag::NamedColorList(NamedColorListRef::from_ptr(aligned_mut(data))),
                DataTag | Ps2CRD0Tag | Ps2CRD1Tag | Ps2CRD2Tag | Ps2CRD3Tag | Ps2CSATag | Ps2RenderingIntentTag => {
                    cast(data).map(Tag::ICCData).unwrap_or(Tag::None)
                }
                AToB0Tag | AToB1Tag | AToB2Tag | BToA0Tag | BToA1Tag | BToA2Tag | DToB0Tag | DToB1Tag | DToB2Tag | DToB3Tag | BToD0Tag | BToD1Tag
                | BToD2Tag | BToD3Tag | GamutTag | Preview0Tag | Preview1Tag | Preview2Tag => Tag::Pipeline(PipelineRef::from_ptr(aligned_mut(data))),
                BlueTRCTag | GrayTRCTag | GreenTRCTag | RedTRCTag => Tag::ToneCurve(ToneCurveRef::from_ptr(aligned_mut(data))),
                ColorimetricIntentImageStateTag => {
                    read_u32_enum_tag::<ffi::ColorimetricIntentImageState>(data).map(Tag::ColorimetricIntentImageState).unwrap_or(Tag::None)
                }
                PerceptualRenderingIntentGamutTag | SaturationRenderingIntentGamutTag => {
                    read_u32_enum_tag::<Intent>(data).map(Tag::Intent).unwrap_or(Tag::None)
                }
                TechnologyTag => read_u32_enum_tag::<ffi::TechnologySignature>(data).map(Tag::Technology).unwrap_or(Tag::None),
                ViewingConditionsTag => {
                    if check_enum_field!(data.cast::<ffi::ICCViewingConditions>(), ffi::ICCViewingConditions, IlluminantType, ffi::IlluminantType) {
                        cast(data).map(Tag::ICCViewingConditions).unwrap_or(Tag::None)
                    } else {
                        Tag::None
                    }
                }
                MeasurementTag => {
                    if check_enum_field!(
                        data.cast::<ffi::ICCMeasurementConditions>(),
                        ffi::ICCMeasurementConditions,
                        IlluminantType,
                        ffi::IlluminantType
                    ) {
                        cast(data).map(Tag::ICCMeasurementConditions).unwrap_or(Tag::None)
                    } else {
                        Tag::None
                    }
                }
                ProfileSequenceDescTag | ProfileSequenceIdTag => cast(data)
                    .filter(|seq: &&ffi::SEQ| check_enum_field!(seq.seq, ffi::PSEQDESC, technology, ffi::TechnologySignature))
                    .map(Tag::SEQ)
                    .unwrap_or(Tag::None),
                ScreeningTag => {
                    let ch = data.add(mem::offset_of!(ffi::Screening, Channels)).cast::<ffi::ScreeningChannel>();
                    let ok = (0..16).all(|i| check_enum_field!(ch.add(i), ffi::ScreeningChannel, SpotShape, ffi::SpotShape));
                    if ok {
                        cast(data)
                            .map(|s: &ffi::Screening| {
                                // typecheck for unsafe above
                                let _: &[ffi::ScreeningChannel; 16] = &s.Channels;
                                Tag::Screening(s)
                            })
                            .unwrap_or(Tag::None)
                    } else {
                        Tag::None
                    }
                }
                UcrBgTag => cast(data).map(Tag::UcrBg).unwrap_or(Tag::None),
                VcgtTag => {
                    let ptrs = data.cast::<*mut u8>(); // array of pointers
                    Tag::VcgtCurves([
                        ToneCurveRef::from_ptr(aligned_mut(*ptrs)),
                        ToneCurveRef::from_ptr(aligned_mut(*ptrs.add(1))),
                        ToneCurveRef::from_ptr(aligned_mut(*ptrs.add(2))),
                    ])
                }
                CicpTag => cast(data).map(Tag::VideoSignal).unwrap_or(Tag::None),
                MHC2Tag => cast(data).map(Tag::MHC2).unwrap_or(Tag::None),
                _ => Tag::None,
            }
        }
    }
}

#[test]
fn tone_curves_tag() {
    let mut icc = Profile::new_srgb();
    let _ = icc.read_tag(TagSignature::VcgtTag);

    icc.remove_tag(TagSignature::ProfileDescriptionTag);

    let mut desc = MLU::new(1);
    desc.set_text("Test ICC", Locale::none());
    icc.write_tag(TagSignature::ProfileDescriptionTag, Tag::MLU(&desc));

    let tc = ToneCurve::new(2.0);
    let tc_refs: [&ToneCurveRef; 3] = [&tc, &tc, &tc];
    let before = Tag::VcgtCurves(tc_refs);
    icc.write_tag(TagSignature::VcgtTag, before);
    let before = Tag::VcgtCurves(tc_refs);
    let after = icc.read_tag(TagSignature::VcgtTag);

    match (before, after) {
        (Tag::VcgtCurves(a), Tag::VcgtCurves(b)) => {
            assert_eq!(a[0].estimated_entries(), b[0].estimated_entries());
            assert_eq!(a[1].estimated_entries(), b[1].estimated_entries());
            assert_eq!(a[2].estimated_entries(), b[2].estimated_entries());
        }
        _ => panic!("bad tags"),
    }
    icc.icc().unwrap();
}
