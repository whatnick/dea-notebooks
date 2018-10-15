HOME=/g/data/if87/ARD_interoperability/L2
DATA=/g/data/if87/ARD_interoperability/L2

for i in `ls -1 $HOME/unzip`; do if [ ! -d $HOME/yamls/$i ]; then  mkdir -p $HOME/yamls/$i ; fi; done
for i in `ls $HOME/yamls/`; do cp ledaps_lasrc_prepare.sh $HOME/qsub_scripts/$i.qsub; sed -i -e "s/TARGET/$i/g" "$HOME/qsub_scripts/$i.qsub"; echo 'qsub $HOME/qsub_scripts/'$i'.qsub'; done
