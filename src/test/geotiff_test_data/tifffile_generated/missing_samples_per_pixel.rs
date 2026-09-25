//! A grayscale image that omits SamplesPerPixel (tag 277), as PerkinElmer Phenix images do.

use crate::tags::PhotometricInterpretation;
use crate::test::util::open_tiff;

#[tokio::test]
async fn test_missing_samples_per_pixel_defaults_to_one() {
    let (_reader, tiff) =
        open_tiff("geotiff-test-data/tifffile_generated/fixtures/missing_samples_per_pixel.tif")
            .await;
    let ifd = &tiff.ifds()[0];

    assert_eq!(ifd.samples_per_pixel(), 1);
    assert_eq!((ifd.image_width(), ifd.image_height()), (64, 64));
    assert_eq!(
        ifd.photometric_interpretation(),
        PhotometricInterpretation::BlackIsZero
    );
    assert_eq!(ifd.strip_offsets().map(|offsets| offsets.len()), Some(1));
}
