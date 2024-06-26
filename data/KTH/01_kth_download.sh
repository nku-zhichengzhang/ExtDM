TARGET_DIR=$1
if [ -z $TARGET_DIR ]
then
  echo "Must specify target directory"
else
  mkdir $TARGET_DIR/raw
  for c in walking jogging running handwaving handclapping boxing
  do  
    URL=http://www.csc.kth.se/cvap/actions/"$c".zip
    wget $URL -P $TARGET_DIR/raw
    mkdir $TARGET_DIR/raw/$c
    unzip $TARGET_DIR/raw/"$c".zip -d $TARGET_DIR/raw/$c
    rm $TARGET_DIR/raw/"$c".zip
  done

fi