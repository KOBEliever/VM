import pandas as pd
import numpy as np

def _init_wma(df: pd.DataFrame) ->pd.DataFrame:
    counter_col = [
        'P2_PAD_WAFER_CNT_20042', 
        'P2_HEAD_RR_WAFER_CNT_20038', 
        'P1_PAD_WAFER_CNT_10042', 
        'P1_HEAD_RR_WAFER_CNT_10038'
    ]

    for i in range(1,len(df)):
        if all(pd.isna(df.loc[i, col]) for col in counter_col):
            continue

        start_idx = max(0, i-6)

        for j in range(start_idx, i):
            diff = None
            for col in counter_col:
                val_i = df.loc[i, col]
                val_j = df.loc[j, col]

                if pd.isna(val_i) or pd.isna(val_j):
                    continue

                diff = val_i - val_j
                if 1 <= diff <= 6 and diff == int(diff):
                    diff = int(diff)
                    break
                else:
                    diff = None
            
            if diff is not None:
                col_name = f'PST_GLB_WMA_PRE_THK_{diff}_{80002 + diff}'
                if pd.pd.isna(df.loc[i, col_name]):
                    df.loc[i, col_name] = df.loc[j, 'PST_GLB_CMP_THK_80000']

    return df

def _cal_wma_features(wma_data: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
    try:
        weight = np.array()
        time_column = "P1_X_TIME"
        time_threshold_min = 5
        cnt_threshold_min = 1.0
        counter_col = [
        'P2_PAD_WAFER_CNT_20042', 
        'P2_HEAD_RR_WAFER_CNT_20038', 
        'P1_PAD_WAFER_CNT_10042', 
        'P1_HEAD_RR_WAFER_CNT_10038'
        ]
        wma_col = ['PST_GLB_WMA_PRE_THK_1_80003'
                   'PST_GLB_WMA_PRE_THK_2_80004',
                   'PST_GLB_WMA_PRE_THK_3_80005',
                   'PST_GLB_WMA_PRE_THK_4_80006',
                   'PST_GLB_WMA_PRE_THK_5_80007',
                   'PST_GLB_WMA_PRE_THK_6_80008']
        data = _init_wma_columns(data)
        columns_list = data.columns.to_list()
        wma_id_list = []
        if wma_data is not None and not wma_data.empty and not data.empty:
            wma_id_list = wma_data["WAFER_ID"].to_list()
            wma_data = wma_data[columns_list]
            replace_dict = {col: {None: np.nan} for col in wma_data.columns}
            wma_data = wma_data.replace(replace_dict)
            data = pd.concat([wma_data, data], ignore_index=True)
        data = _init_wma(data)

        if time_column in data.columns:
            # TODO
            data[time_column] = pd.to_datetime(data[time_column])
        else:
            print("no time column")
            return data[~data["WAFER_ID"].isin(wma_id_list) if wma_id_list else data]

        data = data.sort_values(time_column)

        for eqp, group in data.groupby('PROC_EQP'):
            group = group.sort_values(time_column)

            for i in range(7, len(group) + 1):
                window = group.iloc[i - 7: i]
                current_idx = group.index[i - 1]

                time_diff = window[time_column].diff().dropna()

                if (time_diff > time_threshold_min).any():
                    continue

                has_spike = False
                for col in counter_col:
                    if col in window.columns:
                        diff = window[col].diff().abs()
                        if (diff > cnt_threshold_min).any():
                            has_spike = True
                            break
                
                if has_spike:
                    continue
                
                window = window.iloc[6:]

                thk_values = window['PST_GLB_CMP_THK_80000'].values
                if len(thk_values) == len(weight):
                    weighted_avg = np.dot(thk_values, weight) / np.sum(weight)
                    data.at[current_idx, 'PST_GLB_WMA_THK_80002'] = weighted_avg
                    for j, col in enumerate(wma_col):
                        data.at[current_idx, col] = window.iloc[5-j]['PST_GLB_CMP_THK_80000']
        mask = ~data["WAFER_ID"].isin(wma_id_list)
        data = data[mask]
        return data
    
    except Exception as e:
        print(e)
        return data
