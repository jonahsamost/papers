zip -r files.zip *.py *.sh
scp -P "$2" files.zip root@"$1":~/files.zip
rm files.zip
