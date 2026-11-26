- `Load Component` errors for single files now explain that the file only holds weights, name the
  models checked for a config and say how to supply one. A file that does not match its config is
  reported as such: the error names the config used and where it came from, and tells you to set
  `Config (Optional)` to the repo id or `config.json` of the model the file was made from.
  [#43](https://github.com/griptape-ai/griptape-nodes-library-diffusers/issues/43)
