# Configuration

Copy `runtime.env.example` to `runtime.env` and set the model, LIBERO, and
optional runtime paths. `protocol.json` identifies the DMT evaluation protocol and is not
a result file.

`scene/base_scene.template.xml` is a source template. Generate a local scene
with `bash bin/prepare_scene.sh` after installing LIBERO assets. The generated
scene is ignored by git.
