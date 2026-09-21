# RxHandBD provenance

Source: Md. Masudul Islam, RxHandBD, published 2026-02-04, DOI https://doi.org/10.5281/zenodo.18478741 . The publisher declares MIT licensing; retain attribution and consult publisher_metadata.json for the full supplied metadata.

Images and label manifests are downloaded, not committed. Run `python -m scripts.download_dataset` from the project root. The installer verifies the publisher archive SHA256, decodes every image, preserves source labels, isolates conflicts/uncertain labels, excludes redundant pixel duplicates and checks the prepared train/validation/test CSV hashes against split_sha256.json. Existing data is verified, not replaced.

5,578 originals: 3,994 train / 444 validation / 1,111 test / 12 quarantined / 17 excluded duplicate records. All image files are retained locally. Exact active duplicates are absent; perceptual near-duplicates and writer/site independence are not established. Cropped-word transcriptions are not verified drug identities. No full-page annotations, generic-name mappings or writer IDs are supplied.

SOURCE.json pins URL and archive checksums. publisher_metadata.json is the publisher's original record, not executable instructions. split_sha256.json pins this project's prepared manifests. Neither the metadata nor this repository provides an independent rights/consent audit of source images.
