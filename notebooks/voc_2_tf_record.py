from datasets import pascalvoc_to_tfrecords

pascalvoc_to_tfrecords.run('../../PascalVOC/dataset/pascalvoc2007/VOCtest_06-Nov-2007/', '.', 'voc_test')