# llama.cpp: the MIP token channel patch

`mip-channel.patch` applies to llama.cpp at commit `1deefcca395743049c3820ab8f9b15043f3e9446` (https://github.com/ggml-org/llama.cpp,
MIT). On this PC it is the branch `mip-channel` of `C:/AI/llama.cpp` (commits `704279085`, `eae8e9a01`), built by `scripts/pc/build-w4a8-mip.cmd` into
`build-w4a8-mip` (sm_89 only, the flags of `build-w4a8`). Build label: **4B NVFP4 W4A8 · llama.cpp+MIP · PC layer**.

What it adds (design: `docs/plans/2026-11-12-mip-token-channel-design.md`):
- `common/mip-channel.{h,cpp}`: a `llama_sampler` at the head of the sampling chain that streams every accepted token of a request to the
  MIP service (loopback socket, `MIP_CHANNEL=host:port`) and applies the plans it returns (only the tokens of a value's tokenizations
  allowed at a step; a ban of digits for one step after it). Off unless the request carries `mip_id`; with the service unreachable or a
  plan late (`MIP_CHANNEL_MS`, default 50) the request goes on unsteered. An exception inside the shim is logged and turns that request's
  shim passive.
- `common_params_sampling::mip_id` and the server request field `mip_id` (`tools/server/server-schema.cpp`).
- `ws2_32` linked into `llama-common` on Windows.

To apply elsewhere: `git -C <llama.cpp> checkout 1deefcca3 && git -C <llama.cpp> apply <this dir>/mip-channel.patch`.

## llama.cpp's licence

MIT License

Copyright (c) 2023-2026 The ggml authors

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR
ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH
THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
