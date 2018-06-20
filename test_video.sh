# Make prediction from mp4 video file (ffmpeg is required)
#python test_video.py --video_file sample_data/juggling.mp4 --rendered_output sample_data/predicted_video.mp4 --weight pretrain/TRN_moments_RGB_InceptionV3_TRNmultiscale_segment8_best.pth.tar --arch InceptionV3 --dataset moments

# Make prediction with input a a folder name with RGB frames
python test_video.py --modality RGB --arch BNInception --test_segments 8 --consensus_type TRNmultiscale --dataset jester --weight pretrain/TRN_jester_RGB_BNInception_TRNmultiscale_segment8_cls5_best.pth.tar --frame_folder sample_data/juggling_frames
