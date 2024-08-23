import pandas as pd
import numpy as np
import os
import argparse 
import logging
import warnings 

warnings.filterwarnings("ignore")

class Preprocess():
    def __init__(self, csv_file, csv_save_file, partial=False):

        self.csv_file = os.path.join(os.getcwd(), csv_file)
        print("self.csv_file", self.csv_file)
        if csv_save_file != "":
            self.save = True
        else:
            self.save = False
        self.df = pd.read_csv(self.csv_file)
        self.df = self.df.sort_values(by="Dataset_prefix_group_id")
        if partial:
            self.df = self.df.iloc[:partial]
        self.min_seq_length, self.max_seq_length = self.get_min_max_seq_lengths()

        self.sequences, self.bboxes = self.get_sequence_bbox()
        self.df_prepro = self.df.copy()
        self.csv_save_file = csv_save_file

       
    def get_min_max_seq_lengths(self):
        grouped_data = self.df['Dataset_prefix_group_id'].value_counts()
        return min(grouped_data), max(grouped_data)
        
    def pad(self, length_target, min_length, max_length, mode="last"):
        if mode == "last": 
            self.remove_or_split(length_target)

            for group_id in self.df_prepro.Dataset_prefix_group_id.unique():
                sequence = self.df_prepro[self.df_prepro["Dataset_prefix_group_id"] == group_id]
                
                if len(sequence.index) < min_length:
                    self.remove_group_sequence(group_id)
                    continue

                if len(sequence.index) > length_target:
                    continue

                self.add_sequence(sequence, length_target, group_id)


    def trunc(self, length_target, max_length, min_length, mode="first"): 
        if mode == "first": 
            self.remove_or_split(length_target)

            for group_id in self.df_prepro.Dataset_prefix_group_id.unique():
                sequence = self.df_prepro[self.df_prepro["Dataset_prefix_group_id"] == group_id]

                if len(sequence.index) > max_length or len(sequence.index) < length_target:
                    self.remove_group_sequence(group_id)
                    continue

                # remove first images
                else:
                    self.remove_sequence(sequence, length_target, group_id)
                   
                
    def remove_or_split(self, length_target): 
        self.length_target = length_target
        _ = self.df_prepro.groupby('Dataset_prefix_group_id').apply(self.split)
        self.df_prepro['Dataset_prefix_group_id'] = _.values

        return self.df_prepro
        
    def split(self, group):
        num_subgroups = np.ceil(len(group)/self.length_target).astype(int)
        new_group_ids = []
        for i in range(num_subgroups):
            count = min(self.length_target, len(group) - i * self.length_target)
            new_group_ids.extend([f"{group.name}_{i+1}"] * count)
        
        return pd.Series(new_group_ids, index=group.index)

    def split2(self, group, length_target):
        num_subgroups = np.ceil(len(group)/length_target).astype(int)
        new_group_ids = []
        for i in range(num_subgroups):
            count = min(length_target, len(group) - i * length_target)
            new_group_ids.extend([f"{group.name}_{i+1}"] * count)
        
        return pd.Series(new_group_ids, index=group.index)



    def remove_group_sequence(self, group_id): 
        self.df_prepro = self.df_prepro.loc[self.df_prepro["Dataset_prefix_group_id"] != group_id]
        

    def add_sequence(self, sequence, length_target, group_id): 
        while len(sequence.index) < length_target:
            index = sequence.index[-1]
            self.df_prepro  = pd.concat([self.df_prepro, pd.DataFrame([self.df_prepro.loc[index]])], ignore_index=True)
            sequence = self.df_prepro[self.df_prepro["Dataset_prefix_group_id"] == group_id]
        

    def remove_sequence(self, sequence, length_target, group_id): 
        while len(sequence.index) > length_target:
            index = sequence.index[0]
            self.df_prepro = self.df_prepro.drop(index, ignore_index=True)
            sequence = self.df_prepro[self.df_prepro["Dataset_prefix_group_id"] == group_id]


    
    def preprocess_sequence(self, option="padding", pad_mode="last", trunc_mode="first",  min_length=1, max_length=5):
        if option=="padding":
            self.pad(self.max_seq_length if self.max_seq_length < max_length else max_length, min_length, max_length, pad_mode)
        
        if option=="truncating":
            self.trunc(self.min_seq_length if self.min_seq_length > min_length else min_length, min_length, max_length, trunc_mode)

        if self.save:
            self.df_prepro.to_csv(self.csv_save_file)

        
        
    def get_sequence_bbox(self, max_nan=1.0):
        df = self.df.sort_values(by="Dataset_prefix_group_id")
        sequences = df.groupby('Dataset_prefix_group_id')['Rel_Image_Path'].apply(list).tolist()
        bboxes_xcenter  = df.groupby('Dataset_prefix_group_id')['yolo_bbox_xcenter'].apply(list).tolist()
        bboxes_ycenter  = df.groupby('Dataset_prefix_group_id')['yolo_bbox_ycenter'].apply(list).tolist()
        bboxes_width  = df.groupby('Dataset_prefix_group_id')['yolo_bbox_width'].apply(list).tolist()
        bboxes_height  = df.groupby('Dataset_prefix_group_id')['yolo_bbox_height'].apply(list).tolist()

        assert len(sequences) == len(bboxes_xcenter) == len(bboxes_ycenter) == len(bboxes_width) == len(bboxes_height)

        listes_filtrees_1 = []
        listes_filtrees_2 = []
        listes_filtrees_3 = []
        listes_filtrees_4 = []
        seq_filtrees = []

        for sub_seq, sub_list_1, sub_list_2, sub_list_3, sub_list_4 in zip(sequences, bboxes_xcenter, bboxes_ycenter, bboxes_width, bboxes_height):
            pourcentage_1 = self.pourcentage_nan(sub_list_1)
            pourcentage_2 = self.pourcentage_nan(sub_list_2)
            pourcentage_3 = self.pourcentage_nan(sub_list_3)
            pourcentage_4 = self.pourcentage_nan(sub_list_4)

            if pourcentage_1 < max_nan and pourcentage_2 < max_nan and pourcentage_3 < max_nan and pourcentage_4 < max_nan:
                listes_filtrees_1.append(sub_list_1)
                listes_filtrees_2.append(sub_list_2)
                listes_filtrees_3.append(sub_list_3)
                listes_filtrees_4.append(sub_list_4)
                seq_filtrees.append(sub_seq)
                
        bboxes = {"bboxes_xcenter": listes_filtrees_1, 
            "bboxes_ycenter": listes_filtrees_2, 
            "bboxes_width": listes_filtrees_3, 
            "bboxes_height": listes_filtrees_4}
        
        return seq_filtrees, bboxes

    def interpolate_group(self, group): 
        group[self.cols_to_interpolate] = group[self.cols_to_interpolate].interpolate(method=self.method, axis=0, limit_direction="both")
        group[self.cols_to_interpolate] = group[self.cols_to_interpolate].fillna(method="bfill").fillna(method="ffill")
        return group

    
    def preprocess_bbox(self, method="nearest", sigma=1):
        self.method = method
        self.cols_to_interpolate = ["yolo_bbox_xcenter","yolo_bbox_ycenter","yolo_bbox_width","yolo_bbox_height"]
        self.df_prepro = self.df.groupby("Dataset_prefix_group_id").apply(self.interpolate_group)
        self.df_prepro = self.purge_groups_with_all_nans()

        if self.save:
            self.df_prepro.to_csv(self.csv_save_file)

    def purge_groups_with_all_nans(self): 
        self.df_prepro.reset_index(drop=True, inplace=True)
        nans_counts = self.df_prepro.groupby("Dataset_prefix_group_id")[self.cols_to_interpolate].apply(lambda x: x.isna().sum())
        group_sizes = self.df_prepro.groupby("Dataset_prefix_group_id").size()
        group_to_remove = nans_counts.apply(lambda x:(x==group_sizes[x.name]).any(), axis=1)
        group_to_remove = group_to_remove[group_to_remove].index

        return self.df_prepro[~self.df_prepro["Dataset_prefix_group_id"].isin(group_to_remove)]

        
     #TODO : generate_sequence : 
     # gros écarts de temps, on divise les séq en 2 : changer le préfix 
     # FP : au minimum 1 bbox 

    @staticmethod   
    def pourcentage_nan(liste):
        if len(liste) == 0:
            return 0
        nb_nan = np.isnan(liste).sum()
        return nb_nan / len(liste)

        # truncating : on prend le plus petit, et les plus haut on les cut (True)
        # padding (True) : on rajoute des images ou non, on peut rajouter des 0 ou des images jusqu'à arriver au plus grand

        # min_length : on supprime du dataset (si qq est de 2 images, on le supprime si min_length=3)
        # max_length : on supprime du dataset 

        # seq_length (option): mix of "truncating", "padding" 

        # option : comment on rajoute/supprime des images 
        #     - linéaire : si on a 30 images et qu'on en choisit 10, on prend toutes les 3 
        #     - polynomial (ordre): répartition spéciale selon l'ordre (non linéalire) 
        #     - aléatoire 
        #     - gaussian : les extrêmes ne sont pas les bonnes images, on choisit plus d'élements au centre de la distribution 
        #     - norme infini : on prend tous les 0 d'abord puis tous les 1, ou on prend 1 seul truc

        # split (true/false): si on a 2 x plus d'images que seq_length, on renvoit 2 sequences 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess sequences with padding or truncating.")
    parser.add_argument("-f", "--file", type=str, default="pyronear_ds_03_2024_train_val_DS_fp_temporal_dataset.csv",
                        help="Csv file.")
    parser.add_argument("-s", "--save_file", type=str, default="",
                        help="Preprocessed csv file.")
    parser.add_argument("--option", type=str, default="padding", choices=["padding", "truncating"],
                        help="Choose the preprocessing method: padding or truncating.")
    parser.add_argument("--pad_mode", type=str, default="last", choices=["last", "first"],
                        help="Padding mode: add padding at the 'last' or 'first'.")
    parser.add_argument("--trunc_mode", type=str, default="first", choices=["first", "last"],
                        help="Truncating mode: remove elements from the 'first' or 'last'.")
    parser.add_argument("--min_length", type=int, default=1,
                        help="Minimum length to use for padding/truncating.")
    parser.add_argument("--max_length", type=int, default=5,
                        help="Maximum length to use for padding/truncating.")
    parser.add_argument("--log_file", type=str, default="",
                        help="Path to the log file. Logs to console if not specified.")
    parser.add_argument("--log_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Set the logging level.")
    
    args = parser.parse_args()

    if args.log_file:
        logging.basicConfig(filename=args.log_file, level=getattr(logging, args.log_level),
                            format='%(asctime)s - %(levelname)s - %(message)s')
    else:
        logging.basicConfig(level=getattr(logging, args.log_level),
                            format='%(asctime)s - %(levelname)s - %(message)s')

    logging.info("Starting preprocessing...")
    sequence_processor = Preprocess(args.file, args.save_file)

    sequence_processor.preprocess_sequence(option=args.option, pad_mode=args.pad_mode, trunc_mode=args.trunc_mode,
                                           min_length=args.min_length, max_length=args.max_length)
    print(sequence_processor.length_target)
    logging.info("Preprocessing completed successfully.")

    #TODO: analyse rapide du dataframe 
    # nb de groupes, nb d'images
    # logging.warn
