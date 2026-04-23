# clean pathData_20

import pandas as pd
def clean_data(input_path, output_path):
    df = pd.read_csv(input_path)
    print(f"Original data size: {len(df)}")
    pos_x = df['pos_x'].values
    pos_y = df['pos_y'].values
    target_x_0 = df['target_x_0'].values
    target_y_0 = df['target_y_0'].values

    # if distance between current position and first target point is more than 1.5, then remove the data point
    distance = ((pos_x - target_x_0) ** 2 + (pos_y - target_y_0) ** 2) ** 0.5
    df_cleaned = df[distance <= 1.5]
    df_cleaned.to_csv(output_path, index=False)
    print(f"Cleaned data size: {len(df_cleaned)}")


def remove_straight(input_path, output_path):
    df = pd.read_csv(input_path)
    obs_flags = [col for col in df.columns if 'obstacle_flag' in col]
    has_obstacle = df[obs_flags].sum(axis=1) > 0
    df_obstacles = df[has_obstacle]
    df_boring = df[~has_obstacle]

    df_boring_sampled = df_boring.sample(frac=0.10, random_state=42)

    df_balanced = pd.concat([df_obstacles, df_boring_sampled]).sample(frac=1.0)

    print(f"Original shape: {df.shape}")
    print(f"Balanced shape: {df_balanced.shape}")
    df_balanced.to_csv(output_path, index=False)


if __name__ == "__main__":
    input_path = '../pathData_20_5.csv'
    output_path = '../pathData_20_5_cleaned_2.csv'
    # clean_data(input_path, output_path)
    remove_straight(input_path, output_path)