- Fixed `Generate Media Latents` applying the wrong strength when LTX-2 HDR IC-LoRA was given several
  video references of different lengths. Shorter references are now padded to the longest by repeating
  their last frame, and a log warning names each padded video. An image reference mixed with a longer
  video stops with an error, because an image can't be padded without becoming a video.
