TARGET_DIR=$1
if [ -z $TARGET_DIR ]
then
  echo "Must specify target directory"
else
#   mkdir $TARGET_DIR/
#   URL=https://www.cs.ubc.ca/~wsgh/fdm/carla/no-traffic.tar.gz
#   wget $URL -P $TARGET_DIR
  tar -zxvf $TARGET_DIR/no-traffic.tar.gz -C $TARGET_DIR
fi

# Example:
# cd /home/ubuntu/zzc/vidpred/edm-neurips23/data/CARLA_Town_01
# bash 01_carla_town_01_download.sh /mnt/hdd/zzc/data/video_prediction/CARLA_Town_01