SDTBS(Software Defined TBS)
===========================
- This project aims to support simulation on real GPU device.
- TB preemption functionality added

## Build

- You need to install cuda SDK or cuda related packages.
- `# ./configure --with-cuda=<cuda home path>`
- \# make

## Running

- `# cd src/sdtbs`
- `# ./sdtbs <options> <benchmark spec>...`
- you can check detailed options with ./sdtbs -h 
