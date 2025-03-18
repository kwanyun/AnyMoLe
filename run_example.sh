CHAR_NAME=${1:-Amy}
MOTION_NAME=${2:-Amy_careful_Walking_input}

cd ./anymole-icadapt
python run_video_inference.py --frames_dir ../anymole-render/images/"$CHAR_NAME"/"$MOTION_NAME" --text_input --ICAdapt --interp --onlykey 750 --stage 2

cd ../anymole-render
python pose3d_train.py --char_name "$CHAR_NAME" --motion_name "$MOTION_NAME"
python motion_video_mimicking.py --char_name "$CHAR_NAME" --motion_path "$MOTION_NAME"
