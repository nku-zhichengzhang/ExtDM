import os
from data.KTH.kth_actions_frames import kth_actions_dict, settings, actions, person_ids, MCVD_person_ids

def convert_with_official_split():
    """
    len train 760
    len valid 768
    len test  863
    min_length train is 30
    min_length valid is 26
    min_length test is 24
    """
    for data_split in ['train', 'valid', 'test']:
        print('Converting ' + data_split)

        with open(f"{data_split}-official.txt", 'w') as f:
            min_length = 1e6
            split_person_ids = person_ids[data_split]
            for person_id in split_person_ids:
                # print('     Converting person' + person_id)
                for action in kth_actions_dict['person'+person_id]:
                    for setting in kth_actions_dict['person'+person_id][action]:
                        for frame_idxs in kth_actions_dict['person'+person_id][action][setting]:
                            file_name = 'person' + person_id + '_' + action + '_' + setting + '_uncomp.avi'
                            file_path = os.path.join(action, file_name)
                            # index of kth_actions_frames.py starts from 1 but we need 0
                            # and wo should fix to [a,b) not [a,b]
                            # eg: 1-123, 124-345, length is 345
                            # ->  0-122, 123-344, length is 345
                            # ->  0-123, 123-345, length is 345 same
                            start_frame_idxs = frame_idxs[0] - 1
                            end_frames_idxs = frame_idxs[1]

                            min_length = min(min_length, end_frames_idxs - start_frame_idxs)
                            
                            f.write(f"{file_path} {start_frame_idxs} {end_frames_idxs}\n")
        print('min_length', data_split, 'is', min_length)
        print('Converting', data_split, 'done.')

def convert_with_all_frames():
    """
    len train 191
    len valid 192
    len test  216
    min_length train is 250
    min_length valid is 204
    min_length test is 256
    """
    for data_split in ['train', 'valid', 'test']:
        cnt = 0
        print('Converting ' + data_split)

        with open(f"{data_split}.txt", 'w') as f:
            min_length = 1e6
            split_person_ids = person_ids[data_split]
            for person_id in split_person_ids:
                # print('     Converting person' + person_id)
                for action in kth_actions_dict['person'+person_id]:
                    for setting in kth_actions_dict['person'+person_id][action]:
                        a_list = sorted(kth_actions_dict['person'+person_id][action][setting])
                        # index of kth_actions_frames.py starts from 1 but we need 0
                        # and wo should fix to [a,b) not [a,b]
                        # eg: 1-12, ... 124-345, length is 345
                        # ->  0-11, ... 123-344, length is 345
                        # ->  0-345, length is 345 same
                        start_frame_idxs = a_list[0][0] - 1
                        end_frames_idxs = a_list[-1][1]

                        file_name = 'person' + person_id + '_' + action + '_' + setting + '_uncomp.avi'
                        file_path = os.path.join(action, file_name)
                        min_length = min(min_length, end_frames_idxs - start_frame_idxs)
                            
                        f.write(f"{file_path}\n")
                        cnt += 1
        print('num       ', data_split, 'is', cnt)
        print('min_length', data_split, 'is', min_length)
        print('Converting', data_split, 'done.')
        print("")

def convert_MCVD_setting():
    """
    num        train is 479
    num        valid is 120
    min_length train is 230
    min_length valid is 204
    """

    for data_split in ['train', 'valid']:
        cnt = 0
        print('Converting ' + data_split)

        with open(f"{data_split}.txt", 'w') as f:
            min_length = 1e6
            split_person_ids = MCVD_person_ids[data_split]
            for person_id in split_person_ids:
                # print('     Converting person' + person_id)
                for action in kth_actions_dict['person'+person_id]:
                    for setting in kth_actions_dict['person'+person_id][action]:
                        a_list = sorted(kth_actions_dict['person'+person_id][action][setting])
                        start_frame_idxs = a_list[0][0] - 1
                        end_frames_idxs = a_list[-1][1]

                        file_name = 'person' + person_id + '_' + action + '_' + setting + '_uncomp.avi'
                        file_path = os.path.join(action, file_name)
                        min_length = min(min_length, end_frames_idxs - start_frame_idxs)
                            
                        f.write(f"{file_path}\n")
                        cnt += 1
        print('num       ', data_split, 'is', cnt)
        print('min_length', data_split, 'is', min_length)
        print('Converting', data_split, 'done.')
        print("")

convert_MCVD_setting()

# cd /home/ubuntu15/zzc/code/videoprediction/pred-vdm/data/KTH/
# python 02_kth_train_val_test_split.py