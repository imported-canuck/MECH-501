'''
BearingData.py

This script loads all data files into dictionaries, organized by their RPM
It serves as a helper/baseline script to be imported to the ML models

Load all data files into a dictionary {description}[filename]
Must be run from the same directory as the data files, otherwise modify path
Format: RPM_Fault_Size_LocationSamplerate

RPMs: 1730 (3 hp), 1750 (2 hp), 1772 (1 hp), 1797 (0 hp)
Faults: Ball (B), Inner Raceway (IR), Outer Raceway 3:00 (OR@3), Outer Raceway 6:00 (OR@6), None (Normal)
Size: 0.007", 0.014", 0.021"
Location: Drive End (DE), Fan End (FE)
Samplerate: 12khz, 48khz
'''

files_1730RPM = {
    '1730_B_7_DE12': '1730_B_7_DE12.npz',
    '1730_B_7_DE48': '1730_B_7_DE48.npz',
    '1730_B_7_FE12': '1730_B_7_FE.npz',
    '1730_B_14_DE12': '1730_B_14_DE12.npz',
    '1730_B_14_DE48': '1730_B_14_DE48.npz',
    '1730_B_14_FE12': '1730_B_14_FE.npz',
    '1730_B_21_DE12': '1730_B_21_DE12.npz',
    '1730_B_21_DE48': '1730_B_21_DE48.npz',
    '1730_B_21_FE12': '1730_B_21_FE.npz',

    '1730_IR_7_DE12': '1730_IR_7_DE12.npz',
    '1730_IR_7_DE48': '1730_IR_7_DE48.npz',
    '1730_IR_7_FE12': '1730_IR_7_FE.npz',
    '1730_IR_14_DE12': '1730_IR_14_DE12.npz',
    '1730_IR_14_DE48': '1730_IR_14_DE48.npz',
    '1730_IR_14_FE12': '1730_IR_14_FE.npz',
    '1730_IR_21_DE12': '1730_IR_21_DE12.npz',
    '1730_IR_21_DE48': '1730_IR_21_DE48.npz',
    '1730_IR_21_FE12': '1730_IR_21_FE.npz',

    '1730_Normal': '1730_Normal.npz',
    
    '1730_OR@3_14_FE12': '1730_OR@3_14_FE.npz', # Replacementes for missing OR@6 files
    '1730_OR@3_21_FE12': '1730_OR@3_21_FE.npz',

    '1730_OR@6_7_DE12': '1730_OR@6_7_DE12.npz',
    '1730_OR@6_7_DE48': '1730_OR@6_7_DE48.npz',
    '1730_OR@6_7_FE12': '1730_OR@6_7_FE.npz',
    '1730_OR@6_14_DE12': '1730_OR@6_14_DE12.npz',
    '1730_OR@6_14_DE48': '1730_OR@6_14_DE48.npz',
    '1730_OR@6_21_DE12': '1730_OR@6_21_DE12.npz',
    '1730_OR@6_21_DE48': '1730_OR@6_21_DE48.npz',
}

files_1750RPM = {
    '1750_B_7_DE12': '1750_B_7_DE12.npz',
    '1750_B_7_DE48': '1750_B_7_DE48.npz',
    '1750_B_7_FE12': '1750_B_7_FE.npz',
    '1750_B_14_DE12': '1750_B_14_DE12.npz',
    '1750_B_14_DE48': '1750_B_14_DE48.npz',
    '1750_B_14_FE12': '1750_B_14_FE.npz',    
    '1750_B_21_DE12': '1750_B_21_DE12.npz',
    '1750_B_21_DE48': '1750_B_21_DE48.npz',
    '1750_B_21_FE12': '1750_B_21_FE.npz',

    '1750_IR_7_DE12': '1750_IR_7_DE12.npz',
    '1750_IR_7_DE48': '1750_IR_7_DE48.npz', 
    '1750_IR_7_FE12': '1750_IR_7_FE.npz',
    '1750_IR_14_DE12': '1750_IR_14_DE12.npz',
    '1750_IR_14_DE48': '1750_IR_14_DE48.npz',
    '1750_IR_14_FE12': '1750_IR_14_FE.npz',
    '1750_IR_21_DE12': '1750_IR_21_DE12.npz',
    '1750_IR_21_DE48': '1750_IR_21_DE48.npz',
    '1750_IR_21_FE12': '1750_IR_21_FE.npz',

    '1750_Normal': '1750_Normal.npz',

    '1750_OR@3_14_FE12': '1750_OR@3_14_FE.npz',
    '1750_OR@3_21_FE12': '1750_OR@3_21_FE.npz',

    '1750_OR@6_7_DE12': '1750_OR@6_7_DE12.npz',
    '1750_OR@6_7_DE48': '1750_OR@6_7_DE48.npz',
    '1750_OR@6_7_FE12': '1750_OR@6_7_FE.npz',
    '1750_OR@6_14_DE12': '1750_OR@6_14_DE12.npz',
    '1750_OR@6_14_DE48': '1750_OR@6_14_DE48.npz',
    '1750_OR@6_21_DE12': '1750_OR@6_21_DE12.npz',
    '1750_OR@6_21_DE48': '1750_OR@6_21_DE48.npz',
}

files_1772RPM = {
    '1772_B_7_DE12': '1772_B_7_DE12.npz',
    '1772_B_7_DE48': '1772_B_7_DE48.npz',
    '1772_B_7_FE12': '1772_B_7_FE.npz',
    '1772_B_14_DE12': '1772_B_14_DE12.npz',
    '1772_B_14_DE48': '1772_B_14_DE48.npz',
    '1772_B_14_FE12': '1772_B_14_FE.npz',
    '1772_B_21_DE12': '1772_B_21_DE12.npz',
    '1772_B_21_DE48': '1772_B_21_DE48.npz',
    '1772_B_21_FE12': '1772_B_21_FE.npz',

    '1772_IR_7_DE12': '1772_IR_7_DE12.npz',
    '1772_IR_7_DE48': '1772_IR_7_DE48.npz',
    '1772_IR_7_FE12': '1772_IR_7_FE.npz',
    '1772_IR_14_DE12': '1772_IR_14_DE12.npz',
    '1772_IR_14_DE48': '1772_IR_14_DE48.npz',
    '1772_IR_14_FE12': '1772_IR_14_FE.npz',
    '1772_IR_21_DE12': '1772_IR_21_DE12.npz',
    '1772_IR_21_DE48': '1772_IR_21_DE48.npz',
    '1772_IR_21_FE12': '1772_IR_21_FE.npz',

    '1772_Normal': '1772_Normal.npz',

    '1772_OR@3_14_FE12': '1772_OR@3_14_FE.npz',
    '1772_OR@3_21_FE12': '1772_OR@3_21_FE.npz',

    '1772_OR@6_7_DE12': '1772_OR@6_7_DE12.npz',
    '1772_OR@6_7_DE48': '1772_OR@6_7_DE48.npz',
    '1772_OR@6_7_FE12': '1772_OR@6_7_FE.npz',
    '1772_OR@6_14_DE12': '1772_OR@6_14_DE12.npz',
    '1772_OR@6_14_DE48': '1772_OR@6_14_DE48.npz',
    '1772_OR@6_21_DE12': '1772_OR@6_21_DE12.npz',
    '1772_OR@6_21_DE48': '1772_OR@6_21_DE48.npz',
}

files_1797RPM = {
    '1797_B_7_DE12': '1797_B_7_DE12.npz',  
    '1797_B_7_DE48': '1797_B_7_DE48.npz',
    '1797_B_7_FE12': '1797_B_7_FE.npz',
    '1797_B_14_DE12': '1797_B_14_DE12.npz',
    '1797_B_14_DE48': '1797_B_14_DE48.npz',
    '1797_B_14_FE12': '1797_B_14_FE.npz',    
    '1797_B_21_DE12': '1797_B_21_DE12.npz',
    '1797_B_21_DE48': '1797_B_21_DE48.npz',
    '1797_B_21_FE12': '1797_B_21_FE.npz',    

    '1797_IR_7_DE12': '1797_IR_7_DE12.npz',
    '1797_IR_7_DE48': '1797_IR_7_DE48.npz',
    '1797_IR_7_FE12': '1797_IR_7_FE.npz',
    '1797_IR_14_DE12': '1797_IR_14_DE12.npz',
    '1797_IR_14_DE48': '1797_IR_14_DE48.npz',
    '1797_IR_14_FE12': '1797_IR_14_FE.npz',
    '1797_IR_21_DE12': '1797_IR_21_DE12.npz',
    '1797_IR_21_DE48': '1797_IR_21_DE48.npz',
    '1797_IR_21_FE12': '1797_IR_21_FE.npz',

    '1797_Normal': '1797_Normal.npz',

    '1797_OR@6_7_DE12': '1797_OR@6_7_DE12.npz',
    '1797_OR@6_7_DE48': '1797_OR@6_7_DE48.npz',
    '1797_OR@6_7_FE12': '1797_OR@6_7_FE.npz',
    '1797_OR@6_14_DE12': '1797_OR@6_14_DE12.npz',
    '1797_OR@6_14_DE48': '1797_OR@6_14_DE48.npz',
    '1797_OR@6_14_FE12': '1797_OR@6_14_FE.npz',
    '1797_OR@6_21_DE12': '1797_OR@6_21_DE12.npz',
    '1797_OR@6_21_DE48': '1797_OR@6_21_DE48.npz',
    '1797_OR@6_21_FE12': '1797_OR@6_21_FE.npz',
}
