# %%
def featurize(df: pd.DataFrame):
    """
    Adds new features to dataframe
    """
    
    df['hist_price_diff'] = np.exp(df.loc[:,'prop_log_historical_price']) - df.loc[:,'price_usd']
    df['diff_price'] = df.loc[:,'visitor_hist_adr_usd'] - trainset.loc[:,'price_usd']
    df['diff_starrating'] = df.loc[:,'visitor_hist_starrating'] - df.loc[:,'prop_starrating']
    df['fee_per_person'] = df.loc[:,'price_usd'] * df.loc[:,'srch_room_count']/(df.loc[:,'srch_adults_count'] + df.loc[:,'srch_children_count'])
    df['score2ma'] = df.loc[:,'prop_location_score2']*df.loc[:,'srch_query_affinity_score']
    df['total_price'] = df.loc[:,'price_usd']*df.loc[:,'srch_room_count']
    
    return df.copy()
