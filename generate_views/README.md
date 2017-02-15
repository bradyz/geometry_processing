## Dependencies

Libigl
Png++
Eigen
Glm
Glfw
OpenGL 4.1


## Usage

```bash
mkdir build
cd build
cmake ..
```

At this point you should add the libigl include path (use ccmake).

In the build directory -

```bash
make -j4
```

In the main directory, supply the text file with all the paths to the data, and
the output directory (must include the slash on the end).

All the nested paths must be created, so if you have /path/to/model/mesh.off,
$OUTPUT_DIR/path/to/model must be created before the script runs.

```bash
./build/bin/generate_views $INPUT_TXT $OUTPUT_DIR
```
