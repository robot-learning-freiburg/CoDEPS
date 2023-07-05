# yapf: disable
# pylint: skip-file

from datasets.cityscapes_labels import Label

labels = [
    #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
    Label(  'road'                 ,  7 ,        0 , 'flat'            , 1       , False        , False        , (128, 64,128) ),
    Label(  'sidewalk'             ,  8 ,        1 , 'flat'            , 1       , False        , False        , (244, 35,232) ),
    Label(  'building'             , 11 ,        2 , 'construction'    , 2       , False        , False        , ( 70, 70, 70) ),
    Label(  'fence'                , 13 ,        3 , 'construction'    , 2       , False        , False        , (190,153,153) ),
    Label(  'pole'                 , 17 ,        4 , 'object'          , 3       , False        , False        , (153,153,153) ),
    Label(  'traffic sign'         , 20 ,        5 , 'object'          , 3       , False        , False        , (220,220,  0) ),
    Label(  'vegetation'           , 21 ,        6 , 'nature'          , 4       , False        , False        , (107,142, 35) ),
    Label(  'terrain'              , 22 ,        7 , 'nature'          , 4       , False        , False        , (152,251,152) ),
    Label(  'sky'                  , 23 ,        8 , 'sky'             , 5       , False        , False        , ( 70,130,180) ),
    # ----------------------------------------------------------------
    Label(  'person'               , 24 ,        9 , 'human'           , 6       , True         , False        , (220, 20, 60) ),
    Label(  'rider'                , 25 ,       10 , 'human'           , 6       , True         , False        , (255,  0,  0) ),
    Label(  'car'                  , 26 ,       11 , 'vehicle'         , 7       , True         , False        , (  0,  0,142) ),
    Label(  'truck'                , 27 ,       12 , 'vehicle'         , 7       , True         , False        , (  0,  0, 70) ),
    Label(  'two-wheeler'          , 33 ,       13 , 'vehicle'         , 7       , True         , False        , (119, 11, 32) ),
]
