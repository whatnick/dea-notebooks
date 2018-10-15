HOME=/g/data/if87/ARD_interoperability/L2

for i in `find $HOME/qsub_scripts -name "*.qsub"`; do qsub $i; done

