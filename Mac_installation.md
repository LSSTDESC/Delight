# Install instructions on a Mac

As of OSX Catalina, Apple has dropped built-in `openmp` support for the clang/gcc that ships with most Macs.  In order to successfully build and install Delight, you will need to set your local copy of gcc to work with openmp.  There are a variety of ways to accomplish this, the most straightforward is to use Mac Homebrew to install several packages.  Follow the steps below.

1) Using homebrew, install updated versions of llvm and openmp with the command:

`brew install llvm openmp`

2) update gcc with the command :
`brew install gcc`

3) Homebrew will install gcc to the install directory that you specify, e.g. `/usr/local/Cellar/`, locate the gcc binary in that install path.  It is likely that Homebrew will append the version number to disambiguate from the default gcc already installed, e.g. `gcc-11`.
Set your computer to point to this gcc rather than the default gcc, for example by adding the Homebrew gcc's path to the front of your `$PATH` and aliasing `gcc-11` to `gcc`

4) Run the Delight install as usual with
```
pip install -r requirements.txt
python setup.py build_ext --inplace
python setup.py install
```

This should successfully install Delight on Mac.
