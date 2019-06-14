SDTBS(Software Defined TBS)
===========================
- This project aims to support simulation on real GPU device.

## Build

- You need to install cuda SDK or cuda related packages.
- # ./configure --with-cuda=&lt;cuda home path&gt;
- # make

## Running

- # cd src/sdtbs
- # ./sdtbs &lt;options&gt; &lt;benchmark spec&gt;...
- you can check detailed options with ./sdtbs -h 
