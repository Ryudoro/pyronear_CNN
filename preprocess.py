import pandas as pd
import numpy as np
import os

class Preprocess():
    def __init__(self, csv_file):
        self.csv_file = csv_file
        self.df = pd.read_csv(self.csv_file)
        self.min_seq_length, self.max_seq_length = self.get_min_max_seq_lengths()
        self.sequences, self.bboxes = self.get_sequence_bbox()
       
    def get_min_max_seq_lengths(self):
        grouped_data = self.df['Prefix_group_id'].value_counts()
        return min(grouped_data), max(grouped_data)
        
    def pad(self, sequence: list, length_target, mode="last"): 
        if mode == "last": 
            # duplicate last images
            while len(sequence) < length_target:
                sequence.append(sequence[-1])
        return sequence
    
    def trunc(self, sequence: list, length_target, mode="first"): 
        if mode == "first": 
            # duplicate first images
            while len(sequence) > length_target:
                sequence.pop(0)
            
        return sequence
        
    def preprocess_sequence(self, sequence:list, option="padding", pad_mode="last", trunc_mode="first",  min_length=5, max_length=50):
        if len(sequence) > max_length or len(sequence) < min_length:
            return None 

        if option=="padding":
            sequence = self.pad(sequence, self.max_seq_length if self.max_seq_length < max_length else max_length, pad_mode)
        
        if option=="truncating":
            sequence = self.trunc(sequence, self.min_seq_length if self.min_seq_length > min_length else min_length, trunc_mode)

        return sequence
        
        
    def get_sequence_bbox(self, max_nan=1.0):
        df = self.df.sort_values(by="Prefix")
        sequences = df.groupby('Prefix_group_id')['Image_Path'].apply(list).tolist()
        sequences = df.groupby('Prefix_group_id')['Image_Path'].apply(list).tolist()
        bboxes_xcenter  = df.groupby('Prefix_group_id')['yolo_bbox_xcenter'].apply(list).tolist()
        bboxes_ycenter  = df.groupby('Prefix_group_id')['yolo_bbox_ycenter'].apply(list).tolist()
        bboxes_width  = df.groupby('Prefix_group_id')['yolo_bbox_width'].apply(list).tolist()
        bboxes_height  = df.groupby('Prefix_group_id')['yolo_bbox_height'].apply(list).tolist()

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
        
        print(len(sequences), len(seq_filtrees))
        return seq_filtrees, bboxes
    
    def preprocess_bbox(self, method="nearest", sigma=1):
        bboxes_inter = {"bboxes_xcenter": [], 
            "bboxes_ycenter": [], 
            "bboxes_width": [], 
            "bboxes_height": []}
        for key in self.bboxes.keys():
            print(len(self.bboxes[key]))
            for box in self.bboxes[key]:
                bbox = pd.Series(box)
                bbox = bbox.interpolate(method=method, limit_direction="both")
                bbox = bbox.fillna(method="bfill")
                bbox = bbox.fillna(method="ffill")
                bboxes_inter[key].append(bbox.tolist())
        return bboxes_inter
        
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
    data_dir = "/Users/marguerite/workspace_DS/"
    csv_file = os.path.join(data_dir, "df_DS_fp_newlines_multiple_bbox.csv")
    # csv_file = os.path.join(data_dir, "df_pyronear_ds_03_2024_train_w_datetime_groups.csv")
    # csv_file = os.path.join(data_dir, "df_pyronear_ds_03_2024_val_w_datetime_groups.csv")

    # Testing 
    preprocess = Preprocess(csv_file)
    # print(len(preprocess.preprocess_sequence(sequence, option="truncating")))

    dico = preprocess.preprocess_bbox(method="nearest", sigma=1)
    # print(dico)
