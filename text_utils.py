from sklearn.model_selection import train_test_split

def clean_text(df_col, stopwords):
    # Text cleaning
        df_col = df_col.str.replace("[^a-zA-Z0-9]", " ") # Retain only alphabets
        tokenized_doc = df_col.apply(lambda x: x.split())
        tokenized_doc = tokenized_doc.apply(
            lambda x: [item for item in x if item not in stopwords]
            )
        detokenized_doc = [] # de-tokenization 
        for i in range(len(df_col)): 
            t = ' '.join(tokenized_doc[i]) 
            detokenized_doc.append(t) 

        df_col = detokenized_doc
        return df_col

def data_split(df, seed):
    # Data splits
    train, val = train_test_split(
        df, test_size=0.2, random_state=seed, 
        stratify=df['Category'] # some classes not found in each split
        )
    X_train = train['title'].values
    y_train = train['Category'].values
    X_val = val['title'].values
    y_val = val['Category'].values
    return train, val, X_train, y_train, X_val, y_val