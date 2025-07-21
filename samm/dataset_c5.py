import os

import cv2
import pandas as pd
import torch
import torch.utils.data as data

from GAMDSS import *


class sammDataSet(data.Dataset):
    def __init__(self, raf_path, phase,num_loso, transform = None, basic_aug = False, transform_norm=None):
        self.phase = phase
        self.transform = transform
        self.raf_path = raf_path
        self.transform_norm = transform_norm
        SUBJECT_COLUMN =0
        NAME_COLUMN = 1
        ONSET_COLUMN = 2
        APEX_COLUMN = 3
        OFF_COLUMN = 4
        LABEL_AU_COLUMN = 5
        LABEL_ALL_COLUMN = 6


        df = pd.read_excel(os.path.join(r'D:\PycharmData\SAMM', 'SAMM.xlsx'),usecols=[0,1,3,4,5,8,9])
        df['Subject'] = df['Subject'].apply(str)

        if phase == 'train':
            dataset = df.loc[df['Subject'] != num_loso]
        else:
            dataset = df.loc[df['Subject'] == num_loso]

        Subject = dataset.iloc[:, SUBJECT_COLUMN].values
        File_names = dataset.iloc[:, NAME_COLUMN].values
        Label_all = dataset.iloc[:, LABEL_ALL_COLUMN].values
        Onset_num = dataset.iloc[:, ONSET_COLUMN].values
        Apex_num = dataset.iloc[:, APEX_COLUMN].values
        Offset_num = dataset.iloc[:, OFF_COLUMN].values
        Label_au = dataset.iloc[:, LABEL_AU_COLUMN].values
        self.file_paths_on = []
        self.file_paths_off = []
        self.file_paths_apex = []
        self.label_all = []
        self.label_au = []
        self.sub= []
        self.file_names =[]
        a=0
        b=0
        c=0
        d=0
        e=0
        # use aligned images for training/testing
        for (f,sub,onset,apex,offset,label_all,label_au) in zip(File_names,Subject,Onset_num,Apex_num,Offset_num,Label_all,Label_au):


            # if label_all == 'Anger' or label_all == 'Contempt' or label_all == 'Disgust' or label_all == 'Fear' or label_all == 'Sadness' or label_all == 'Happiness' or label_all == 'Surprise':
            if label_all == 'Anger' or label_all == 'Contempt' or label_all == 'Happiness' or label_all == 'Surprise' or label_all == 'Other':
                self.file_paths_on.append(onset)
                self.file_paths_off.append(offset)
                self.file_paths_apex.append(apex)
                self.sub.append(sub)
                self.file_names.append(f)
                if label_all == 'Anger':
                    self.label_all.append(0)
                    a=a+1
                elif label_all == 'Contempt':
                    self.label_all.append(1)
                    b=b+1
                elif label_all == 'Happiness':
                    self.label_all.append(2)
                    c=c+1
                elif label_all == 'Surprise':
                    self.label_all.append(3)
                    d=d+1
                else:
                    self.label_all.append(4)
                    e=e+1

            # label_au =label_au.split("+")
                if isinstance(label_au, int):
                    self.label_au.append([label_au])
                else:
                    label_au = label_au.split("+")
                    self.label_au.append(label_au)

            ##label

        self.basic_aug = basic_aug
        #self.aug_func = [image_utils.flip_image,image_utils.add_gaussian_noise]

    def __len__(self):
        return len(self.file_paths_on)

    def __getitem__(self, idx):
        ##sampling strategy for training set
        if self.phase == 'train':
            onset = self.file_paths_on[idx]
            apex = self.file_paths_apex[idx]
            offset = self.file_paths_off[idx]
            sub = str(self.sub[idx])
            f = str(self.file_names[idx])

            # on0, apex0, off0 = samm_full_difference(int(onset), int(apex), int(offset), self.raf_path, sub, f)
            # on0 = str(on0)
            # apex0 = str(apex0)
            # offset = str(off0)
            on0 = str(onset)
            apex0 = str(apex)
            offset = str(offset)

        else:##sampling strategy for testing set
            onset = self.file_paths_on[idx]
            apex = self.file_paths_apex[idx]
            offset = self.file_paths_off[idx]
            sub = str(self.sub[idx])
            f = str(self.file_names[idx])
            on0 = str(onset)
            apex0 = str(apex)
            offset = str(offset)


        on0 = '%03d' % int(sub) + '_'+ on0+'.jpg'

        apex0 = '%03d' % int(sub) + '_' + apex0 + '.jpg'

        offset = '%03d' % int(sub) + '_' + offset + '.jpg'

        path_on0 = os.path.join(self.raf_path, '%03d' % int(sub), f, on0)

        path_apex0 = os.path.join(self.raf_path, '%03d' % int(sub), f, apex0)

        path_off0 = os.path.join(self.raf_path, '%03d' % int(sub), f, offset)

        image_on0 = cv2.imread(path_on0)

        image_apex0 = cv2.imread(path_apex0)

        image_off0 = cv2.imread(path_off0)

        image_on0 = image_on0[:, :, ::-1] # BGR to RGB

        image_apex0 = image_apex0[:, :, ::-1]

        image_off0 = image_off0[:, :, ::-1]

        label_all = self.label_all[idx]
        label_au = self.label_au[idx]

        # normalization for testing and training
        if self.transform is not None:
            image_on0 = self.transform(image_on0)

            image_apex0 = self.transform(image_apex0)

            image_off0 = self.transform(image_off0)

            ALL = torch.cat(
                (image_on0, image_apex0, image_off0), dim=0)
            ## data augmentation for training only
            if self.transform_norm is not None and self.phase == 'train':
                ALL = self.transform_norm(ALL)
            image_on0 = ALL[0:3, :, :]

            image_apex0 = ALL[3:6, :, :]

            image_off0 = ALL[6:9, :, :]


            return image_on0, image_apex0, image_off0, label_all

class samm_max_DataSet(data.Dataset):
    def __init__(self, raf_path, phase,num_loso, transform = None, basic_aug = False, transform_norm=None):
        self.phase = phase
        self.transform = transform
        self.raf_path = raf_path
        self.transform_norm = transform_norm
        SUBJECT_COLUMN =0
        NAME_COLUMN = 1
        ONSET_COLUMN = 2
        APEX_COLUMN = 3
        OFF_COLUMN = 4
        LABEL_AU_COLUMN = 5
        LABEL_ALL_COLUMN = 6


        df = pd.read_excel(os.path.join(r'D:\PycharmData\SAMM', 'SAMM.xlsx'),usecols=[0,1,3,4,5,8,9])
        df['Subject'] = df['Subject'].apply(str)

        if phase == 'train':
            dataset = df.loc[df['Subject'] != num_loso]
        else:
            dataset = df.loc[df['Subject'] == num_loso]

        Subject = dataset.iloc[:, SUBJECT_COLUMN].values
        File_names = dataset.iloc[:, NAME_COLUMN].values
        Label_all = dataset.iloc[:, LABEL_ALL_COLUMN].values
        Onset_num = dataset.iloc[:, ONSET_COLUMN].values
        Apex_num = dataset.iloc[:, APEX_COLUMN].values
        Offset_num = dataset.iloc[:, OFF_COLUMN].values
        Label_au = dataset.iloc[:, LABEL_AU_COLUMN].values
        self.file_paths_on = []
        self.file_paths_off = []
        self.file_paths_apex = []
        self.label_all = []
        self.label_au = []
        self.sub= []
        self.file_names =[]
        a=0
        b=0
        c=0
        d=0
        e=0
        # use aligned images for training/testing
        for (f,sub,onset,apex,offset,label_all,label_au) in zip(File_names,Subject,Onset_num,Apex_num,Offset_num,Label_all,Label_au):


            # if label_all == 'Anger' or label_all == 'Contempt' or label_all == 'Disgust' or label_all == 'Fear' or label_all == 'Sadness' or label_all == 'Happiness' or label_all == 'Surprise':
            if label_all == 'Anger' or label_all == 'Contempt' or label_all == 'Happiness' or label_all == 'Surprise' or label_all == 'Other':
                self.file_paths_on.append(onset)
                self.file_paths_off.append(offset)
                self.file_paths_apex.append(apex)
                self.sub.append(sub)
                self.file_names.append(f)
                if label_all == 'Anger':
                    self.label_all.append(0)
                    a=a+1
                elif label_all == 'Contempt':
                    self.label_all.append(1)
                    b=b+1
                elif label_all == 'Happiness':
                    self.label_all.append(2)
                    c=c+1
                elif label_all == 'Surprise':
                    self.label_all.append(3)
                    d=d+1
                else:
                    self.label_all.append(4)
                    e=e+1

            # label_au =label_au.split("+")
                if isinstance(label_au, int):
                    self.label_au.append([label_au])
                else:
                    label_au = label_au.split("+")
                    self.label_au.append(label_au)

            ##label

        self.basic_aug = basic_aug
        #self.aug_func = [image_utils.flip_image,image_utils.add_gaussian_noise]

    def __len__(self):
        return len(self.file_paths_on)

    def __getitem__(self, idx):
        ##sampling strategy for training set
        if self.phase == 'train':
            onset = self.file_paths_on[idx]
            apex = self.file_paths_apex[idx]
            offset = self.file_paths_off[idx]
            sub = str(self.sub[idx])
            f = str(self.file_names[idx])

            on0, apex0, off0 = samm_min_max_difference(int(onset), int(apex), int(offset), self.raf_path, sub, f)
            on0 = str(on0)
            apex0 = str(apex0)
            offset = str(off0)
            # on0 = str(onset)
            # apex0 = str(apex)
            # offset = str(offset)

        else:##sampling strategy for testing set
            onset = self.file_paths_on[idx]
            apex = self.file_paths_apex[idx]
            offset = self.file_paths_off[idx]
            sub = str(self.sub[idx])
            f = str(self.file_names[idx])
            on0 = str(onset)
            apex0 = str(apex)
            offset = str(offset)


        on0 = '%03d' % int(sub) + '_'+ on0+'.jpg'

        apex0 = '%03d' % int(sub) + '_' + apex0 + '.jpg'

        offset = '%03d' % int(sub) + '_' + offset + '.jpg'

        path_on0 = os.path.join(self.raf_path, '%03d' % int(sub), f, on0)

        path_apex0 = os.path.join(self.raf_path, '%03d' % int(sub), f, apex0)

        path_off0 = os.path.join(self.raf_path, '%03d' % int(sub), f, offset)

        image_on0 = cv2.imread(path_on0)

        image_apex0 = cv2.imread(path_apex0)

        image_off0 = cv2.imread(path_off0)

        image_on0 = image_on0[:, :, ::-1] # BGR to RGB

        image_apex0 = image_apex0[:, :, ::-1]

        image_off0 = image_off0[:, :, ::-1]

        label_all = self.label_all[idx]
        label_au = self.label_au[idx]

        # normalization for testing and training
        if self.transform is not None:
            image_on0 = self.transform(image_on0)

            image_apex0 = self.transform(image_apex0)

            image_off0 = self.transform(image_off0)

            ALL = torch.cat(
                (image_on0, image_apex0, image_off0), dim=0)
            ## data augmentation for training only
            if self.transform_norm is not None and self.phase == 'train':
                ALL = self.transform_norm(ALL)
            image_on0 = ALL[0:3, :, :]

            image_apex0 = ALL[3:6, :, :]

            image_off0 = ALL[6:9, :, :]


            return image_on0, image_apex0, image_off0, label_all


class max_samm_DataSet(data.Dataset):
    def __init__(self, raf_path, phase,num_loso, transform = None, basic_aug = False, transform_norm=None):
        self.phase = phase
        self.transform = transform
        self.raf_path = raf_path
        self.transform_norm = transform_norm
        SUBJECT_COLUMN = 0
        NAME_COLUMN = 1
        ONSET_COLUMN = 2
        APEX_COLUMN = 3
        OFF_COLUMN = 4
        LABEL_ALL_COLUMN = 5
        val_onset_column = 6
        val_apex_column = 7


        df = pd.read_excel(os.path.join(r'D:\PycharmData\SAMM', 'max_samm.xlsx'),usecols=[0,1,2,3,4,5,6,7])
        df['Subject'] = df['Subject'].apply(str)

        if phase == 'train':
            dataset = df.loc[df['Subject'] != num_loso]
        else:
            dataset = df.loc[df['Subject'] == num_loso]

        Subject = dataset.iloc[:, SUBJECT_COLUMN].values
        File_names = dataset.iloc[:, NAME_COLUMN].values
        Label_all = dataset.iloc[:, LABEL_ALL_COLUMN].values
        Onset_num = dataset.iloc[:, ONSET_COLUMN].values
        Apex_num = dataset.iloc[:, APEX_COLUMN].values
        Offset_num = dataset.iloc[:, OFF_COLUMN].values
        val_onset = dataset.iloc[:, val_onset_column].values
        val_apex = dataset.iloc[:, val_apex_column].values
        self.file_paths_on = []
        self.file_paths_off = []
        self.file_paths_apex = []
        self.val_onsets = []
        self.val_apexs = []
        self.label_all = []
        self.label_au = []
        self.sub= []
        self.file_names =[]
        a=0
        b=0
        c=0
        d=0
        e=0
        # use aligned images for training/testing
        for (f,sub,onset,apex,offset,label_all, valonset, valapex) in zip(File_names,Subject,Onset_num,Apex_num,Offset_num,Label_all, val_onset, val_apex):


            # if label_all == 'Anger' or label_all == 'Contempt' or label_all == 'Disgust' or label_all == 'Fear' or label_all == 'Sadness' or label_all == 'Happiness' or label_all == 'Surprise':
            if label_all == 'Anger' or label_all == 'Contempt' or label_all == 'Happiness' or label_all == 'Surprise' or label_all == 'Other':
                self.file_paths_on.append(onset)
                self.file_paths_off.append(offset)
                self.file_paths_apex.append(apex)
                self.val_onsets.append(valonset)
                self.val_apexs.append(valapex)
                self.sub.append(sub)
                self.file_names.append(f)
                if label_all == 'Anger':
                    self.label_all.append(0)
                    a=a+1
                elif label_all == 'Contempt':
                    self.label_all.append(1)
                    b=b+1
                elif label_all == 'Happiness':
                    self.label_all.append(2)
                    c=c+1
                elif label_all == 'Surprise':
                    self.label_all.append(3)
                    d=d+1
                else:
                    self.label_all.append(4)
                    e=e+1


            ##label

        self.basic_aug = basic_aug
        #self.aug_func = [image_utils.flip_image,image_utils.add_gaussian_noise]

    def __len__(self):
        return len(self.file_paths_on)

    def __getitem__(self, idx):
        ##sampling strategy for training set
        if self.phase == 'train':
            onset = self.file_paths_on[idx]
            apex = self.file_paths_apex[idx]
            offset = self.file_paths_off[idx]
            sub = str(self.sub[idx])
            f = str(self.file_names[idx])

            # on0, apex0, off0 = samm_max_difference(int(onset), int(apex), int(offset), self.raf_path, sub, f)
            # on0 = str(on0)
            # apex0 = str(apex0)
            # offset = str(off0)
            on0 = str(onset)
            apex0 = str(apex)
            offset = str(offset)

        else:##sampling strategy for testing set
            onset = self.val_onsets[idx]
            apex = self.val_apexs[idx]
            offset = self.file_paths_off[idx]
            sub = str(self.sub[idx])
            f = str(self.file_names[idx])
            on0 = str(onset)
            apex0 = str(apex)
            offset = str(offset)


        on0 = '%03d' % int(sub) + '_'+ on0+'.jpg'

        apex0 = '%03d' % int(sub) + '_' + apex0 + '.jpg'

        offset = '%03d' % int(sub) + '_' + offset + '.jpg'

        path_on0 = os.path.join(self.raf_path, '%03d' % int(sub), f, on0)

        path_apex0 = os.path.join(self.raf_path, '%03d' % int(sub), f, apex0)

        path_off0 = os.path.join(self.raf_path, '%03d' % int(sub), f, offset)

        image_on0 = cv2.imread(path_on0)

        image_apex0 = cv2.imread(path_apex0)

        image_off0 = cv2.imread(path_off0)

        image_on0 = image_on0[:, :, ::-1] # BGR to RGB

        image_apex0 = image_apex0[:, :, ::-1]

        image_off0 = image_off0[:, :, ::-1]

        label_all = self.label_all[idx]


        # normalization for testing and training
        if self.transform is not None:
            image_on0 = self.transform(image_on0)

            image_apex0 = self.transform(image_apex0)

            image_off0 = self.transform(image_off0)

            ALL = torch.cat(
                (image_on0, image_apex0, image_off0), dim=0)
            ## data augmentation for training only
            if self.transform_norm is not None and self.phase == 'train':
                ALL = self.transform_norm(ALL)
            image_on0 = ALL[0:3, :, :]

            image_apex0 = ALL[3:6, :, :]

            image_off0 = ALL[6:9, :, :]


            return image_on0, image_apex0, image_off0, label_all