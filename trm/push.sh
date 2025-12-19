zip -r files.zip . -i "*.py" "*.sh" "*.txt"
scp -P "$2" files.zip root@"$1":~/files.zip
rm files.zip
