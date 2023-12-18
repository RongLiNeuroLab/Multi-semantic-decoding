from datasets.vgdataset import VGDataset
#from datasets.cocodataset import CoCoDataset
from datasets.voc07dataset import Voc07Dataset
from datasets.voc12dataset import Voc12Dataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
def get_train_test_set(fMRI_dir,fMRI_train_list,fMRI_test_list, train_label, test_label,fMRI_length,args = None):
    print('You will perform multi-scale on images for scale 256')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    scale_size = args.scale_size
    crop_size = args.crop_size
    num_class = args.num_classes
    train_data_transform = transforms.Compose([transforms.Resize((scale_size, scale_size)),
                                               transforms.RandomChoice([transforms.RandomCrop(256),
                                               transforms.RandomCrop(248)]),
                                               transforms.Resize((crop_size, crop_size)),
                                               transforms.ToTensor(),
                                               normalize])
    
    test_data_transform = transforms.Compose([transforms.Resize((scale_size, scale_size)),
                                             transforms.CenterCrop(crop_size),
                                             transforms.ToTensor(),
                                             normalize])

    if  args.dataset == 'VG':
        train_set = VGDataset(fMRI_dir,fMRI_train_list,train_data_transform, train_label,num_class,fMRI_length)
        test_set = VGDataset(fMRI_dir,fMRI_test_list, test_data_transform, test_label,num_class,fMRI_length)
# =============================================================================
#     elif args.dataset == 'VOC2007':
#         train_set = Voc07Dataset(train_dir, train_anno, train_data_transform, train_label)
#         test_set = Voc07Dataset(test_dir, test_anno, test_data_transform, test_label)
#     elif args.dataset == 'VOC2012':
#         train_set = Voc12Dataset(train_dir, train_anno, train_data_transform, train_label)
#         test_set = Voc12Dataset(test_dir, test_anno, test_data_transform, test_label)
# =============================================================================

    train_loader = DataLoader(dataset=train_set,
                              num_workers=args.workers,
                              batch_size=args.batch_size,
                              shuffle = True)
    test_loader = DataLoader(dataset=test_set, 
                              num_workers=args.workers,
                              batch_size=args.batch_size,
                              shuffle = False)
    return train_loader, test_loader, train_set.__len__(), test_set.__len__()
