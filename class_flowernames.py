import json

path = 'data/imagelabels.mat'


def label_class_map(path):
    """

    :param path: path of images label numbers
    :return: dictionary with keys class labels and values flower names
    """
    # TODO : see how you generalize this function so that you it generates my dict for you.
    label_dict = scipy.io.loadmat('data/imagelabels.mat')
    label_list = label_dict['labels'][0]
    labels = ['pink primrose', 'hard-leaved pocket orchid', 'canterbury bells', 'sweet pea', 'english marigold',
              'tiger lily', 'moon orchid', 'bird of paradise', 'monkshood', 'globe thistle', 'snapdragon',
              "colt's foot", 'king protea', 'spear thistle', 'yellow iris', 'globe-flower', 'purple coneflower',
              'peruvian lily', 'balloon flower', 'giant white arum lily', 'fire lily', 'pincushion flower',
              'fritillary', 'red ginger', 'grape hyacinth', 'corn poppy', 'prince of wales feathers',
              'stemless gentian', 'artichoke', 'sweet william', 'carnation', 'garden phlox', 'love in the mist',
              'mexican aster', 'alpine sea holly', 'ruby-lipped cattleya', 'cape flower', 'great masterwort',
              'siam tulip', 'lenten rose', 'barbeton daisy', 'daffodil', 'sword lily', 'poinsettia', 'bolero deep blue',
              'wallflower', 'marigold', 'buttercup', 'oxeye daisy', 'common dandelion', 'petunia', 'wild pansy',
              'primula', 'sunflower', 'pelargonium', 'bishop of llandaff', 'gaura', 'geranium', 'orange dahlia',
              'pink-yellow dahlia?', 'cautleya spicata', 'japanese anemone', 'black-eyed susan', 'silverbush',
              'californian poppy', 'osteospermum', 'spring crocus', 'bearded iris', 'windflower', 'tree poppy',
              'gazania', 'azalea', 'water lily', 'rose', 'thorn apple', 'morning glory', 'passion flower', 'lotus',
              'toad lily', 'anthurium', 'frangipani', 'clematis', 'hibiscus', 'columbine', 'desert-rose', 'tree mallow',
              'magnolia', 'cyclamen ', 'watercress', 'canna lily', 'hippeastrum ', 'bee balm', 'ball moss', 'foxglove',
              'bougainvillea', 'camellia', 'mallow', 'mexican petunia', 'bromelia', 'blanket flower', 'trumpet creeper',
              'blackberry lily']

    unique = [int(n) for n in list(set(label_list))]

    unique[:] = [i-1 for i in unique]

    mydict = {}

    for key, value in zip(unique,labels):
        mydict[key] = value

    with open('data/new_class_label_map', 'w') as f:
        json.dump(mydict, f, sort_keys=True, indent=4)

    return mydict