To compile model.py with Pyinstaller.

1. Create a Conda environment:
> conda create --name Territory --file conda-requirements.txt

2. Run command:
> pyinstaller -F --exclude-module PyQt5 --add-data vcruntime140.dll;. --add-data vcomp140.dll;. --hidden-import="sklearn.neighbors._typedefs" --hidden-import="sklearn.neighbors._dist_metrics" --hidden-import="sklearn.neighbors._ball_tree" --hidden-import="sklearn.utils._cython_blas" --hidden-import="sklearn.neighbors._quad_tree" --hidden-import="sklearn.tree._utils" model.py

3. Find created exe-file in the dist folder
