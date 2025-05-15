import pandas as pd
import os

def process_nsl_kdd_train_only(train_input_path, output_path):
    if not os.path.exists(train_input_path):
        print(f"Error: The file at {train_input_path} does not exist.")
        return

    train_df = pd.read_csv(train_input_path, header=None)
    column_names = [
        'duration','protocol_type','service','flag','src_bytes','dst_bytes','land',
        'wrong_fragment','urgent','hot','num_failed_logins','logged_in','num_compromised',
        'root_shell','su_attempted','num_root','num_file_creations','num_shells','num_access_files',
        'num_outbound_cmds','is_host_login','is_guest_login','count','srv_count','serror_rate',
        'srv_serror_rate','rerror_rate','srv_rerror_rate','same_srv_rate','diff_srv_rate',
        'srv_diff_host_rate','dst_host_count','dst_host_srv_count','dst_host_same_srv_rate',
        'dst_host_diff_srv_rate','dst_host_same_src_port_rate','dst_host_srv_diff_host_rate',
        'dst_host_serror_rate','dst_host_srv_serror_rate','dst_host_rerror_rate','dst_host_srv_rerror_rate',
        'attack_type','difficulty_level'
    ]
    train_df.columns = column_names
    train_df['attack_type'] = train_df['attack_type'].apply(lambda x: 'normal' if x == 'normal.' else 'attack')
    train_df = train_df.drop(columns=['num_outbound_cmds', 'is_host_login', 'difficulty_level'], errors='ignore')

    for col in train_df.select_dtypes(include=['float64', 'int64']).columns:
        train_df[col] = (train_df[col] - train_df[col].min()) / (train_df[col].max() - train_df[col].min())

    os.makedirs(output_path, exist_ok=True)
    train_output_path = os.path.join(output_path, 'processed_train.csv')
    train_df.to_csv(train_output_path, index=False)
    print(f"Processed training data saved at: {train_output_path}")

if __name__ == "__main__":
    train_input_path = "/home/manas/threat_detection/data/raw/KDDTrain+.txt"
    output_path = "/home/manas/threat_detection/data/processed/"
    process_nsl_kdd_train_only(train_input_path, output_path)