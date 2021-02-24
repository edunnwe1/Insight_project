def sipakmed_to_csv(s_path):

    """
        This function creates a single .csv file from individual .dat files
        Inputs: containing folder string
    """
    import pandas as pd
    from os import listdir
    
    sipakmed_df = pd.DataFrame() #initialize empty dataframe

    for file in listdir(s_path)[0:]:
        if file.endswith(".dat"):
            df2 = pd.read_table(s_path+file, sep=',',header=None)
            # add features
            if 'NUC' in file:
                df2[28] = 'NUC'
            else:
                df2[28] = 'nuc'
            df2[29] = file[0].lower()
            if file[0].lower() in 'sp':
                df2[30] = 1
            else:
                df2[30] = 0
            # concatenate
            sipakmed_df = pd.concat([sipakmed_df,df2])

 # create headers
    sipakmed_df.columns = ['cluster_id','image_id','area','major_axis_length','minor_axis_length','eccentricity','orientation','equivalent_diameter','solidity','extent','meanI_R','meanC_R','smooth_R','moment-3-R','uniformity-R','entropy-R','meanI_G','meanC_G','smooth_G','moment-3-G','uniformity-G','entropy-G','meanI_B','meanC_B','smooth_B','moment-3-B','uniformity-B','entropy-B','Nucleus/Cytoplasm','Class','Normal']

        
# save to file 
    sipakmed_df.to_csv('../data/processed/Sipakmed_existing_database.csv')
    
    return sipakmed_df