## New submission

This is a new package.

## Test environments

* local Ubuntu 24.04, R 4.6.1
* GitHub Actions: ubuntu-latest, macos-latest (R release)
* win-builder: R-devel, R-release

## R CMD check results

0 errors | 0 warnings | 1 note

* New submission.

## Notes for the reviewer

* The model weights (about 400 MB) are not bundled. `download_n3d()`
  fetches them from Hugging Face into the 'hfhub' cache only when the user
  calls it, after asking for consent in interactive sessions; nothing is
  downloaded on load, in examples or in tests.
* Examples need those weights, so they are wrapped in `\donttest{}` and
  guarded by `n3d_exists()`; on a machine without the weights they do
  nothing.
* Tests that need the weights run only under `tinytest::at_home()`.
