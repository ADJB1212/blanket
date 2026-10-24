"""ICC profile conversion and soft proofing through LittleCMS.

Transforms accept 8-bit L, RGB, RGBA, CMYK, and LAB images when the profiles
support the requested modes. Only the current frame is transformed.
"""

from __future__ import annotations

import os
from enum import IntEnum, IntFlag
from math import isfinite
from typing import Protocol

from . import Image
from ._blanket import _cms_create_profile, _cms_profile_info, _CmsTransform


class PyCMSError(Exception):
    """An invalid profile, transform, or color management argument."""


class Intent(IntEnum):
    """Rendering intents for colors outside the destination profile's gamut.

    PERCEPTUAL preserves visual relationships; RELATIVE_COLORIMETRIC maps
    white points and clips out-of-gamut colors; SATURATION favors vividness;
    ABSOLUTE_COLORIMETRIC preserves source colorimetry, including its white.
    """

    PERCEPTUAL = 0
    RELATIVE_COLORIMETRIC = 1
    SATURATION = 2
    ABSOLUTE_COLORIMETRIC = 3


class Flags(IntFlag):
    """LittleCMS transform options that can be combined with bitwise OR.

    NONE uses engine defaults. NOCACHE disables the one-pixel cache and
    NOOPTIMIZE disables optimization. NULLTRANSFORM requests an identity
    transform. SOFTPROOFING enables proof simulation; GAMUTCHECK marks colors
    outside the proofing gamut. BLACKPOINTCOMPENSATION adapts black points,
    and NOWHITEONWHITEFIXUP disables the white-on-white correction.
    HIGHRESPRECALC and LOWRESPRECALC select lookup-table accuracy and memory
    tradeoffs. COPY_ALPHA requests alpha copying; Blanket preserves RGBA
    alpha independently of this flag.
    """

    NONE = 0
    NOCACHE = 0x0040
    NOOPTIMIZE = 0x0100
    NULLTRANSFORM = 0x0200
    GAMUTCHECK = 0x1000
    SOFTPROOFING = 0x4000
    BLACKPOINTCOMPENSATION = 0x2000
    NOWHITEONWHITEFIXUP = 0x0004
    HIGHRESPRECALC = 0x0400
    LOWRESPRECALC = 0x0800
    COPY_ALPHA = 0x04000000


class _Readable(Protocol):
    def read(self) -> bytes: ...


class ImageCmsProfile:
    """A validated ICC profile with its original serialized data.

    Attributes:
        filename: Source path, or None for streams and copied profiles.
        profile: This profile object, provided for Pillow-shaped access.

    Examples:
        ```python
        from io import BytesIO
        from blanket import ImageCms

        profile = ImageCms.createProfile("sRGB")
        reopened = ImageCms.ImageCmsProfile(BytesIO(profile.tobytes()))
        print(ImageCms.getProfileDescription(reopened))
        ```
    """

    def __init__(self, profile: str | bytes | os.PathLike[str] | _Readable | ImageCmsProfile) -> None:
        """Load and validate a profile through LittleCMS.

        Args:
            profile: A filesystem path, binary stream at the profile's start,
                or an existing ImageCmsProfile. Bytes are interpreted as a
                path; wrap serialized ICC bytes in io.BytesIO. Caller-owned
                streams are read from their current position and left open.

        Raises:
            PyCMSError: The source cannot be read or its ICC data is invalid.
        """
        try:
            self.filename: str | bytes | None = None
            if isinstance(profile, ImageCmsProfile):
                self._data = profile._data
            elif isinstance(profile, (str, bytes, os.PathLike)):
                self.filename = os.fspath(profile)
                with open(profile, "rb") as stream:
                    self._data = stream.read()
            else:
                self._data = profile.read()
            self._description, self._manufacturer, self._model, self._copyright = _cms_profile_info(self._data)
        except (OSError, ValueError, TypeError, AttributeError) as error:
            raise PyCMSError(str(error)) from error
        self.profile = self

    def tobytes(self) -> bytes:
        """Return the original ICC bytes for embedding or saving.

        Returns:
            Serialized ICC data suitable for an image's info["icc_profile"]
            entry or the icc_profile save argument.
        """
        return self._data

    @property
    def profile_description(self) -> str:
        """The profile's English description, or an empty string if absent."""
        return self._description


type _Profile = str | bytes | os.PathLike[str] | _Readable | ImageCmsProfile


def getOpenProfile(profileFilename: _Profile) -> ImageCmsProfile:
    """Load an ICC profile from a path, binary stream, or existing profile.

    Args:
        profileFilename: A source accepted by ImageCmsProfile.

    Returns:
        A validated ImageCmsProfile.

    Raises:
        PyCMSError: The source cannot be read or parsed as an ICC profile.
    """
    return ImageCmsProfile(profileFilename)


def _profile(profile: _Profile) -> ImageCmsProfile:
    return profile if isinstance(profile, ImageCmsProfile) else ImageCmsProfile(profile)


def createProfile(colorSpace: str, colorTemp: float = 0) -> ImageCmsProfile:
    """Create a built-in color-space profile without an external ICC file.

    Adobe RGB uses D65 and gamma 563/256. Display P3 uses D65 and the sRGB curve.

    Args:
        colorSpace: "sRGB", "Adobe RGB", "Display P3", "LAB", or "XYZ".
            Adobe RGB also accepts "AdobeRGB" and "Adobe RGB (1998)";
            Display P3 also accepts "display P3" and "display-p3".
        colorTemp: LAB white-point temperature in kelvin, from 4000 to 25000.
            Zero selects D50. Other color spaces ignore this value, but it
            must still be finite.

    Returns:
        A validated profile that can be passed directly to transform builders
        or serialized with tobytes().

    Raises:
        PyCMSError: The color space or temperature is invalid, or profile
            creation fails.
    """
    from io import BytesIO

    aliases = {"AdobeRGB": "Adobe RGB", "Adobe RGB (1998)": "Adobe RGB", "display P3": "Display P3", "display-p3": "Display P3"}
    try:
        temperature = float(colorTemp)
        if not isfinite(temperature):
            raise PyCMSError("color temperature must be finite")
        return ImageCmsProfile(BytesIO(_cms_create_profile(aliases.get(colorSpace, colorSpace), temperature)))
    except (ValueError, TypeError, OverflowError) as error:
        raise PyCMSError(str(error)) from error


def getProfileName(profile: _Profile) -> str:
    """Return a human-readable profile name with a trailing newline.

    Args:
        profile: A profile object, filesystem path, or binary stream.

    Returns:
        The model and manufacturer, falling back to the description when
        both are absent. Long model names omit the manufacturer.

    Raises:
        PyCMSError: The profile cannot be loaded.
    """
    value = _profile(profile)
    if not (value._model or value._manufacturer):
        return value._description + "\n"
    if not value._manufacturer or len(value._model) > 30:
        return value._model + "\n"
    return f"{value._model} - {value._manufacturer}\n"


def getProfileDescription(profile: _Profile) -> str:
    """Return the profile description with a trailing newline.

    Args:
        profile: A profile object, filesystem path, or binary stream.

    Returns:
        The English description, or just a newline if the tag is absent.

    Raises:
        PyCMSError: The profile cannot be loaded.
    """
    return _profile(profile)._description + "\n"


def getProfileInfo(profile: _Profile) -> str:
    """Return the profile description and copyright in pyCMS text format.

    Args:
        profile: A profile object, filesystem path, or binary stream.

    Returns:
        Nonempty description and copyright fields separated by two CRLF
        sequences, with two trailing CRLF sequences.

    Raises:
        PyCMSError: The profile cannot be loaded.
    """
    value = _profile(profile)
    return "\r\n\r\n".join(element for element in (value._description, value._copyright) if element) + "\r\n\r\n"


def getProfileCopyright(profile: _Profile) -> str:
    """Return the profile copyright with a trailing newline.

    Args:
        profile: A profile object, filesystem path, or binary stream.

    Returns:
        The English copyright, or just a newline if the tag is absent.

    Raises:
        PyCMSError: The profile cannot be loaded.
    """
    return _profile(profile)._copyright + "\n"


def getProfileManufacturer(profile: _Profile) -> str:
    """Return the profile manufacturer with a trailing newline.

    Args:
        profile: A profile object, filesystem path, or binary stream.

    Returns:
        The English manufacturer text, or just a newline if absent.

    Raises:
        PyCMSError: The profile cannot be loaded.
    """
    return _profile(profile)._manufacturer + "\n"


def getProfileModel(profile: _Profile) -> str:
    """Return the profile device model with a trailing newline.

    Args:
        profile: A profile object, filesystem path, or binary stream.

    Returns:
        The English model text, or just a newline if absent.

    Raises:
        PyCMSError: The profile cannot be loaded.
    """
    return _profile(profile)._model + "\n"


def _intent(value: int) -> int:
    if not isinstance(value, int) or not 0 <= value <= 3:
        raise PyCMSError("rendering intent must be an integer between 0 and 3")
    return int(value)


class ImageCmsTransform:
    """A reusable ICC conversion between fixed profile and pixel mode pairs.

    Transforms operate on the current frame of 8-bit L, RGB, RGBA, CMYK,
    or LAB images with compatible profiles. RGBA output retains input RGBA
    alpha, or receives fully opaque alpha for other input modes. Each result
    receives the output profile in info["icc_profile"].

    Attributes:
        input_profile: Validated source ICC profile.
        output_profile: Validated destination ICC profile.
        input_mode: Required source pixel mode; also exposed as inputMode.
        output_mode: Destination pixel mode; also exposed as outputMode.
    """

    def __init__(
        self,
        input: _Profile,
        output: _Profile,
        input_mode: str,
        output_mode: str,
        intent: int = Intent.PERCEPTUAL,
        proof: _Profile | None = None,
        proof_intent: int = Intent.ABSOLUTE_COLORIMETRIC,
        flags: int = Flags.NONE,
    ) -> None:
        """Build a conversion, optionally simulating a proofing device.

        Args:
            input: Source profile object, filesystem path, or binary stream.
            output: Destination profile object, path, or binary stream.
            input_mode: Source pixel mode supported by the input profile.
            output_mode: Destination mode supported by the output profile.
            intent: Rendering intent from Intent, or an integer from 0 to 3.
            proof: Optional device profile for soft proofing.
            proof_intent: Rendering intent for the proofing device.
            flags: Combined Flags options. Include SOFTPROOFING to simulate
                the proofing device when proof is supplied.

        Raises:
            PyCMSError: Profiles, modes, intents, or flags are invalid, or
                LittleCMS cannot build the requested transform.
        """
        self.input_profile = _profile(input)
        self.output_profile = _profile(output)
        self.input_mode = self.inputMode = input_mode
        self.output_mode = self.outputMode = output_mode
        if not isinstance(flags, int) or not 0 <= flags <= 0xFFFFFFFF:
            raise PyCMSError("flags must be an unsigned 32-bit integer")
        try:
            self.transform = _CmsTransform(
                self.input_profile.tobytes(),
                self.output_profile.tobytes(),
                input_mode,
                output_mode,
                _intent(intent),
                flags,
                None if proof is None else _profile(proof).tobytes(),
                _intent(proof_intent),
            )
        except (ValueError, TypeError, OverflowError) as error:
            raise PyCMSError(str(error)) from error

    def apply(self, image: Image.Image, imOut: Image.Image | None = None) -> Image.Image:
        """Transform the current frame into a new or supplied output image.

        Args:
            image: Open 8-bit image whose mode matches input_mode.
            imOut: Optional open 8-bit destination with matching dimensions
                and output_mode. Passing image applies the transform in place
                and requires identical input and output modes.

        Returns:
            The destination image with the output ICC profile attached. New
            images copy source metadata; supplied images retain their metadata
            apart from the replaced ICC profile.

        Raises:
            PyCMSError: An image is closed, has an incompatible mode, size, or
                sample depth, or the transform cannot be applied.
        """
        if imOut is image:
            return self.apply_in_place(image)
        try:
            if imOut is not None:
                imOut.load()
                if imOut.mode != self.output_mode or imOut.size != image.size or imOut.bit_depth != 8:
                    raise PyCMSError("output image must match transform mode, dimensions, and sample depth")
            native = self.transform.apply(image._native)
        except (ValueError, TypeError) as error:
            raise PyCMSError(str(error)) from error
        result = Image.Image(native) if imOut is None else imOut
        if imOut is not None:
            result._native = native
        else:
            result.info.update(image._metadata_info())
        result.info["icc_profile"] = self.output_profile.tobytes()
        return result

    def apply_in_place(self, image: Image.Image) -> Image.Image:
        """Replace the current frame's pixels and ICC profile in place.

        Args:
            image: Open 8-bit image matching both transform modes. Other
                frames and the current frame's other metadata are retained.

        Returns:
            The same image object, retaining its native core and file format.

        Raises:
            PyCMSError: Input and output modes differ, the image is closed or
                incompatible, or the transform cannot be applied.
        """
        if self.input_mode != self.output_mode:
            raise PyCMSError("in-place transforms require identical input and output modes")
        try:
            self.transform.apply_in_place(image._native)
        except (ValueError, TypeError) as error:
            raise PyCMSError(str(error)) from error
        image.info["icc_profile"] = self.output_profile.tobytes()
        return image


def buildTransform(inputProfile: _Profile, outputProfile: _Profile, inMode: str, outMode: str, renderingIntent: int = Intent.PERCEPTUAL, flags: int = Flags.NONE) -> ImageCmsTransform:
    """Build a reusable ICC conversion between two profiles.

    Args:
        inputProfile: Source profile object, filesystem path, or binary stream.
        outputProfile: Destination profile object, path, or binary stream.
        inMode: Source mode: L, RGB, RGBA, CMYK, or LAB, with 8-bit samples.
        outMode: Destination mode from the same supported set. Each profile
            must support its corresponding mode.
        renderingIntent: Intent enum value or integer from 0 to 3.
        flags: LittleCMS Flags options combined with bitwise OR.

    Returns:
        An ImageCmsTransform for use with applyTransform(). Also available as
        buildTransformFromOpenProfiles.

    Raises:
        PyCMSError: The profiles or arguments are invalid or incompatible.

    Examples:
        ```python
        from blanket import Image, ImageCms

        transform = ImageCms.buildTransform(ImageCms.createProfile("Display P3"), ImageCms.createProfile("sRGB"), "RGB", "RGB")
        image = Image.new("RGB", (8, 8), (80, 140, 220))
        converted = ImageCms.applyTransform(image, transform)
        ```
    """
    return ImageCmsTransform(inputProfile, outputProfile, inMode, outMode, renderingIntent, flags=flags)


def buildProofTransform(
    inputProfile: _Profile,
    outputProfile: _Profile,
    proofProfile: _Profile,
    inMode: str,
    outMode: str,
    renderingIntent: int = Intent.PERCEPTUAL,
    proofRenderingIntent: int = Intent.ABSOLUTE_COLORIMETRIC,
    flags: int = Flags.SOFTPROOFING,
) -> ImageCmsTransform:
    """Build a conversion that simulates a proofing device's color response.

    Args:
        inputProfile: Source profile object, filesystem path, or binary stream.
        outputProfile: Destination profile object, path, or binary stream.
        proofProfile: Profile of the device to simulate, such as a printer.
        inMode: Source mode: L, RGB, RGBA, CMYK, or LAB, with 8-bit samples.
        outMode: Destination mode supported by outputProfile.
        renderingIntent: Source-to-destination rendering intent, from 0 to 3.
        proofRenderingIntent: Rendering intent for the simulated device,
            from 0 to 3; defaults to absolute colorimetric.
        flags: Combined Flags options. SOFTPROOFING is enabled by default;
            retain it when supplying other flags to keep proof simulation.
            GAMUTCHECK can additionally mark out-of-gamut colors.

    Returns:
        A reusable ImageCmsTransform. Also available as
        buildProofTransformFromOpenProfiles.

    Raises:
        PyCMSError: The profiles or arguments are invalid or incompatible.
    """
    return ImageCmsTransform(inputProfile, outputProfile, inMode, outMode, renderingIntent, proofProfile, proofRenderingIntent, flags)


def applyTransform(im: Image.Image, transform: ImageCmsTransform, inPlace: bool = False) -> Image.Image | None:
    """Apply a previously built transform to the current image frame.

    Args:
        im: Open 8-bit image matching the transform's input mode.
        transform: A transform returned by buildTransform() or
            buildProofTransform().
        inPlace: Replace the current frame's pixels and ICC profile. Requires
            identical transform input and output modes; defaults to False.

    Returns:
        A new image with the output ICC profile, or None when inPlace is True.
        A new image copies source metadata. In-place application retains the
        original native core, file format, other metadata, and other frames.

    Raises:
        PyCMSError: The image is closed or incompatible with the transform,
            or in-place application is requested for different modes.
    """
    result = transform.apply_in_place(im) if inPlace else transform.apply(im)
    return None if inPlace else result


def profileToProfile(
    im: Image.Image, inputProfile: _Profile, outputProfile: _Profile, renderingIntent: int = Intent.PERCEPTUAL, outputMode: str | None = None, inPlace: bool = False, flags: int = Flags.NONE
) -> Image.Image | None:
    """Convert the current frame between ICC profiles in a single call.

    Args:
        im: Open 8-bit L, RGB, RGBA, CMYK, or LAB image.
        inputProfile: Source profile object, filesystem path, or binary stream.
        outputProfile: Destination profile object, path, or binary stream.
        renderingIntent: Intent enum value or integer from 0 to 3.
        outputMode: Destination pixel mode; defaults to im.mode. Each profile
            must support its corresponding pixel mode.
        inPlace: Modify im and return None. Requires outputMode to match
            im.mode; defaults to False.
        flags: LittleCMS Flags options combined with bitwise OR.

    Returns:
        A new image, or None for an in-place conversion. The output ICC
        profile replaces info["icc_profile"]. RGBA alpha is preserved. Only
        the current frame is converted; source frame positions do not change.

    Raises:
        PyCMSError: Profiles, modes, intents, or flags are invalid, the image
            is closed or has unsupported sample depth, or conversion fails.

    Examples:
        ```python
        from blanket import Image, ImageCms

        image = Image.new("RGB", (8, 8), (80, 140, 220))
        ImageCms.profileToProfile(image, ImageCms.createProfile("Display P3"), ImageCms.createProfile("sRGB"), inPlace=True)
        ```
    """
    transform = buildTransform(inputProfile, outputProfile, im.mode, outputMode or im.mode, renderingIntent, flags)
    return applyTransform(im, transform, inPlace)


buildTransformFromOpenProfiles = buildTransform
buildProofTransformFromOpenProfiles = buildProofTransform
