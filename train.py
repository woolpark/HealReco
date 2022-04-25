import pandas as pd
import pytorch_lightning as pl

import model


def extract_files():
    data = pd.read_csv('./diabetic_data.csv')
    cat_columns = data.select_dtypes(['object']).columns
    data[cat_columns] = data[cat_columns].apply(lambda x: pd.factorize(x)[0])
    data = data.sort_values('encounter_id').drop_duplicates('patient_nbr', keep='last')
    data = data.drop(['encounter_id', 'patient_nbr'], axis=1)

    train = data.sample(frac=0.7, random_state=42)
    test = data.drop(train.index)
    val = train.sample(frac=0.2, random_state=42)
    train = train.drop(val.index)

    train.to_csv('train.csv')
    test.to_csv('test.csv')
    val.to_csv('val.csv')


def main():
    data = pd.read_csv('./diabetic_data.csv')
    cat_columns = data.select_dtypes(['object']).columns
    data[cat_columns] = data[cat_columns].apply(lambda x: pd.factorize(x)[0])
    data = data.sort_values('encounter_id').drop_duplicates('patient_nbr', keep='last')

    data = data.iloc[0:1000]
    # num_patients = data['patient_nbr'].max() + 1
    num_patients = data.shape[0]
    data = data.drop(['encounter_id', 'patient_nbr'], axis=1)

    train = data.sample(frac=0.7, random_state=42)
    test = data.drop(train.index)
    val = train.sample(frac=0.2, random_state=42)
    train = train.drop(val.index)

    ncf_model = model.NCF(num_patients, train)

    trainer = pl.Trainer(max_epochs=5, reload_dataloaders_every_n_epochs=True, progress_bar_refresh_rate=50,
                         logger=False, checkpoint_callback=False)
    trainer.fit(ncf_model)


if __name__ == "__main__":
    main()