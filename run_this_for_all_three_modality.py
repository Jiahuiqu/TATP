import os

import torch
import numpy as np
import random
import scipy.io as sio
from matplotlib import pyplot as plt
from torch import nn, optim
from operator import truediv
from torch.utils.data import DataLoader
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    classification_report,
    cohen_kappa_score
)
import torch.nn.functional as F

# ####################### #
#    Utility Functions    #
# ####################### #
def set_seed(seed):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def item_to_color(item):
    if item == 0:
        y = np.array([110, 50, 3]) / 255.
    if item == 1:
        y = np.array([147, 67, 46]) / 255.
    if item == 2:
        y = np.array([0, 0, 255]) / 255.
    if item == 3:
        y = np.array([255, 100, 0]) / 255.
    if item == 4:
        y = np.array([0, 255, 123]) / 255.
    if item == 5:
        y = np.array([164, 75, 155]) / 255.
    if item == 6:
        y = np.array([101, 174, 255]) / 255.
    if item == 7:
        y = np.array([118, 254, 172]) / 255.
    if item == 8:
        y = np.array([60, 91, 112]) / 255.
    if item == 9:
        y = np.array([255, 255, 0]) / 255.
    if item == 10:
        y = np.array([255, 255, 125]) / 255.
    if item == 11:
        y = np.array([255, 0, 255]) / 255.
    if item == 12:
        y = np.array([100, 0, 255]) / 255.
    if item == 13:
        y = np.array([0, 172, 254]) / 255.
    if item == 14:
        y = np.array([0, 255, 0]) / 255.
    if item == 15:
        y = np.array([216, 85, 129]) / 255.
    return y

# def item_to_color(item):
#     if item == 0:
#         y = np.array([0, 0, 0]) / 255.0          
#     elif item == 1:
#         y = np.array([59, 132, 70]) / 255.0      
#     elif item == 2:
#         y = np.array([83, 172, 71]) / 255.0      
#     elif item == 3:
#         y = np.array([0, 204, 204]) / 255.0      
#     elif item == 4:
#         y = np.array([146, 82, 52]) / 255.0      
#     elif item == 5:
#         y = np.array([218, 50, 43]) / 255.0      
#     elif item == 6:
#         y = np.array([103, 189, 199]) / 255.0   
#     elif item == 7:
#         y = np.array([229, 229, 240]) / 255.0    
#     elif item == 8:
#         y = np.array([199, 177, 202]) / 255.0    
#     elif item == 9:
#         y = np.array([218, 142, 51]) / 255.0     
#     elif item == 10:
#         y = np.array([224, 220, 83]) / 255.0     
#     elif item == 11:
#         y = np.array([228, 119, 90]) / 255.0    
#     return y

# def item_to_color(item):
#     if item == 0:  
#         y = np.array([0, 0, 0]) / 255.0
#     elif item == 1: 
#         y = np.array([56, 108, 52]) / 255.0  
#     elif item == 2: 
#         y = np.array([228, 55, 38]) / 255.0  
#     elif item == 3:  
#         y = np.array([229, 240, 84]) / 255.0  
#     elif item == 4:  
#         y = np.array([170, 228, 49]) / 255.0  
#     elif item == 5: 
#         y = np.array([139, 222, 38]) / 255.0  
#     elif item == 6:  
#         y = np.array([173, 238, 235]) / 255.0  
#     elif item == 7:  
#         y = np.array([85, 132, 192]) / 255.0  
#     return y

def classification_map(map, save_path):
    fig = plt.figure(frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    fig.add_axes(ax)
    ax.imshow(map)
    #建文件夹
    save_dir = os.path.dirname(save_path)
    fig.savefig(save_path)
    return 0
def acc_reports(y_test, y_pred_test,target_names):
    classification = classification_report(y_test, y_pred_test, digits=2, target_names=target_names)
    oa = accuracy_score(y_test, y_pred_test)
    confusion = confusion_matrix(y_test, y_pred_test)
    each_acc, aa = AA_andEachClassAccuracy(confusion)
    kappa = cohen_kappa_score(y_test, y_pred_test)

    return classification, oa * 100, confusion, each_acc * 100, aa * 100, kappa * 100
def AA_andEachClassAccuracy(confusion_matrix):
    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    return each_acc, average_acc

def calculate_metrics(true_labels, pred_labels):
    """Calculate evaluation metrics"""
    classification_rep = classification_report(
        true_labels, pred_labels,
        digits=2,
        target_names=CLASS_NAMES
    )

    oa = accuracy_score(true_labels, pred_labels)
    conf_matrix = confusion_matrix(true_labels, pred_labels)

    diag = np.diag(conf_matrix)
    class_acc = np.nan_to_num(diag / conf_matrix.sum(axis=1))
    aa = np.mean(class_acc)

    kappa = cohen_kappa_score(true_labels, pred_labels)

    return {
        'report': classification_rep,
        'oa': oa * 100,
        'confusion_matrix': conf_matrix,
        'class_accuracy': class_acc * 100,
        'aa': aa * 100,
        'kappa': kappa * 100
    }


# ####################### #
#    Training Function    #
# ####################### #
def train_model(model, train_loader, criterion, optimizer):
    """Model training routine"""
    model.train()
    for epoch in range(TRAIN_PARAMS['epochs']):
        epoch_loss = 0.0

        for step, (MS_out, HS_out,lidar_out, lable_out, ik_out) in enumerate(train_loader):
            # Move data to device
            MS_out = MS_out.type(torch.float).to(DEVICE)
            lidar_out = lidar_out.type(torch.float).to(DEVICE)
            HS_out = HS_out.type(torch.float).to(DEVICE)
            lable_out = lable_out.type(torch.LongTensor).to(DEVICE) - 1

            text_descriptions = [Class_TEXT[label] for label in lable_out]

            if epoch==0 and step==0:print(lidar_out.shape, HS_out.shape, lable_out.shape)
            outputs, loss_laign, l1 = model(HS_out, lidar_out,MS_out, text_descriptions, text)
            outputs = outputs.type(torch.float).to(DEVICE)
            loss = criterion(outputs, lable_out)
            loss_all = loss + 0.5*loss_laign + 0.1*l1
            # loss_all = loss

            # Backward pass
            optimizer.zero_grad()
            loss_all.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # Print epoch statistics
        avg_loss = epoch_loss / len(train_loader)
        # print("epoch:", epoch, "loss:", loss_all.item(), loss_laign.item(), l1.item(), loss.item())
        print("epoch:", epoch, "loss:", loss.item())

        # Validation every 60 epochs
        if (epoch + 1) % 50 == 0:
            evaluate_model(model,epoch)
            model.train()

# ####################### #
#   Evaluation Function   #
# ####################### #
def evaluate_model(model,epoch):
    """Model evaluation routine"""
    set_seed(SEED)
    model.eval()

    # Initialize data loader
    dataset = LIDARHS(patchsize_inthis, mode='test')
    data_loader = DataLoader(
        dataset,
        batch_size=TRAIN_PARAMS['test_batch_size'],
        shuffle=False,  # Important for evaluation
        num_workers=1
    )

    # Collect predictions and labels
    count = 0
    y_pred_test = 0
    y_test = 0
    with torch.no_grad():
        for step, (MS_out, HS_out,lidar_out, lable_out, ik_out) in enumerate(data_loader):
            # Move data to device
            MS_out = MS_out.type(torch.float).to(DEVICE)
            lidar_out = lidar_out.type(torch.float).to(DEVICE)
            HS_out = HS_out.type(torch.float).to(DEVICE)
            lable_out = lable_out.type(torch.LongTensor).to(DEVICE) - 1

            text_descriptions = ['nothing'] * len(lable_out)

            i = ik_out[:][0]
            k = ik_out[:][1]
            outputs, loss_laign, l1 = model(HS_out, lidar_out,MS_out, text_descriptions, text)
            outputs = np.argmax(outputs.detach().cpu().numpy(), axis=1)

            if count == 0:
                y_pred_test = outputs
                y_test = lable_out.cpu()
                col = i
                nul = k
                count = 1
            else:
                y_pred_test = np.concatenate((y_pred_test, outputs))
                y_test = np.concatenate((y_test, lable_out.cpu()))
                col = np.concatenate((col, i))
                nul = np.concatenate((nul, k))

    map = np.zeros((label_map.shape[0], label_map.shape[1], 3))
    map_pre = np.zeros((label_map.shape[0], label_map.shape[1], 3))
    for p in range(len(y_test)):
        item = y_test[p]+1
        # map[col[p], nul[p]] = item_to_color(item)
        item_pre = y_pred_test[p]+1
        map_pre[col[p], nul[p]] = item_to_color(item_pre)
    # classification_map(map_pre, f'newresult/model_{modelName}_{datasetoff}_batch{epoch}.png')
    # classification_map(map, f'model_{modelName}_gt.png')
    classification, oa, confusion, each_acc, aa, kappa = acc_reports(y_test, y_pred_test, CLASS_NAMES)
    classification = str(classification)
    print(classification)
    print('oa: %.2f' % oa)
    print('aa: %.2f ' % aa)
    print('kappa: %.2f ' % kappa)
    classification = str(classification)
    # file_name = f"result/model_{modelName}_{datasetoff}.txt"
    # file_name = f"newresult/model_{modelName}_{datasetoff}_HSLiDAR.txt"

    with open(file_name, 'a') as x_file:
        x_file.write('{}'.format(prams))
        x_file.write('\n')
        x_file.write('{} Overall accuracy (%)'.format(oa))
        x_file.write('\n')
        x_file.write('{} Average accuracy (%)'.format(aa))
        x_file.write('\n')
        x_file.write('{} Kappa accuracy (%)'.format(kappa))
        x_file.write('\n')
        x_file.write('{} Each accuracy (%)'.format(each_acc))
        x_file.write('\n')
        x_file.write('{}'.format(classification))
        x_file.write('\n')
        x_file.write('{}'.format(confusion))


# ####################### #
#    Main Execution       #
# ####################### #
if __name__ == '__main__':

    # ####################### #
    #      Configurations     #
    # ####################### #
    SEED = 42
    patchsize_inthis = 11


################################################################################################################################################
    text = [
        "HSI records the spectral reflectance or emission characteristics of each spectral band.",
        "LiDAR records the distance or elevation information by measuring the time delay of laser pulses reflected from the target surface.",
        "SAR records the microwave reflectivity characteristics of the surface, capturing high-resolution images regardless of weather or lighting conditions."
    ]

    DEVICE = torch.device('cuda:3')

    from modelALL_any_3_new import modelAll
    model = modelAll(DEVICE, [30, 1, 4],pache_size=patchsize_inthis,num_classes=7).to(DEVICE)

    modelName = 'modelAll_any_3'

    datasetoff = 3 # houston2013 muffl augsburg

    file_name = f"newresult/model_{modelName}_{datasetoff}_guide.txt"
    prams = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(prams)
################################################################################################################################################
    TRAIN_PARAMS = {
        'epochs': 452,
        'batch_size': 64,
        'learning_rate': 1e-4,
        'test_batch_size': 64,
        'save_path': 'best_model.pth',
    }
    if datasetoff == 1:
        from utils.datalodar.dataloader_2013 import LIDARHS

        CLASS_NAMES = ['Healthy grass', 'Stressed grass', 'Synthetic grass', 'Trees'
        , 'Soil', 'Water', 'Residential', 'Commercial', 'Road', 'Highway', 'Railway',
                    'Parking Lot 1', 'Parking Lot 2', 'Tennis Court', 'Running Track']

        Class_TEXT = [
            "The image of Healthy grass, bright green in color, with uniform height.",
            "The image of Stressed grass, dull green or yellowish in color, with uneven height and patchy growth.",
            "The image of Synthetic grass, vibrant green in color, with a uniform and artificial appearance.",
            "The image of Trees, lush green in color, with varying heights and full canopies.",
            "The image of Soil, brown or earthy in color, with a textured and granular surface.",
            "The image of Water, deep blue or reflective in color, smooth and even in texture.",
            "The image of Residential, consisting of individual houses or apartments, with a mix of green spaces and paved areas.",
            "The image of Commercial, featuring large buildings and signage, often with extensive paved areas and parking lots.",
            "The image of Road, gray or black in color, linear in shape, with lane markings and surrounding infrastructure.",
            "The image of Highway, wide and straight, typically with multiple lanes and limited access points.",
            "The image of Railway, long parallel lines with evenly spaced ties, surrounded by cleared land or tracks.",
            "The image of Parking Lot 1, flat and open, with clearly marked parking spaces and vehicle traffic lanes.",
            "The image of Parking Lot 2, similar to Parking Lot 1 but possibly larger or differently arranged.",
            "The image of Tennis Court, rectangular in shape, often green, blue, or red in color, with clear boundary lines.",
            "The image of Running Track, circular in design, typically red or synthetic rubber-colored, with clearly defined lanes."
        ]

        label_map = np.squeeze(
            sio.loadmat('/media/xd132/USER_new/HQG/gqh/ArbilitraryTune_continue/data_2013/All_Label.mat')[
                'All_Label'].astype(np.float32))
    elif datasetoff == 2:#muffl
        from utils.datalodar.dataloader_muufl import LIDARHS

        CLASS_NAMES = [
            'Trees', 'Mostly grass', 'Mix Ground', 'Sand', 'Road', 'Water', 'Building shadow', 'Sidewalk',
            'Curb', 'Cloth panels', 'Buildings'
        ]

        Class_TEXT = [
            "The image of Trees, lush green in color, with varying heights and full canopies.",
            "The image of Mostly grass, bright green in color, covering the majority of the ground with a relatively uniform texture.",
            "The image of Mix Ground, a patchwork of colors and textures, indicating a mixture of soil, grass, and possibly small plants.",
            "The image of Sand, light beige or yellowish in color, with a smooth or slightly rippled surface.",
            "The image of Road, gray or black in color, linear in shape, often with lane markings and surrounding curbs.",
            "The image of Water, deep blue or reflective in color, smooth and even, sometimes showing ripples or waves.",
            "The image of Building shadow, dark areas contrasting with lighter surroundings, outlining the shapes of buildings or structures.",
            "The image of Sidewalk, usually gray or concrete-colored, flat and even, running alongside roads or through pedestrian areas.",
            "The image of Curb, a raised edge along the side of a street or pathway, typically gray or concrete-colored.",
            "The image of Cloth panels, colorful and varied in texture, often found in open areas, used for shading or decorative purposes.",
            "The image of Buildings, diverse in size and shape, ranging from residential homes to commercial structures, casting shadows and occupying significant ground space."
        ]

        label_map = np.squeeze(
            sio.loadmat('/media/xd132/USER_new/HQG/gqh/ArbilitraryTune_continue/data_muufl/All_Label.mat')[
                'All_Label'].astype(np.float32))
    elif datasetoff == 3:#augsburg
        from utils.datalodar.dataloader_agusburg import LIDARHS

        CLASS_NAMES = [
            'Forest', 'Residential', 'Industrial', 'Low Plants', 'Allotment', 'Commercial', 'Water'
        ]

        Class_TEXT = [
            "The image of Forest, a dense expanse with varying shades of green, composed of trees with full canopies and diverse flora.",
            "The image of Residential, an area characterized by homes and apartments, interspersed with gardens, trees, and pathways, showing a mix of construction and green spaces.",
            "The image of Industrial, dominated by large structures and warehouses, often with expansive paved areas, minimal greenery, and signs of activity or storage.",
            "The image of Low Plants, featuring vegetation that is shorter in height, such as shrubs, grasses, or small bushes, presenting a textured but low-profile landscape.",
            "The image of Allotment, divided into neat plots, each potentially growing different types of plants or vegetables, often surrounded by small fences or paths.",
            "The image of Commercial, highlighted by retail stores, offices, and other business establishments, usually accompanied by extensive parking areas and signage.",
            "The image of Water, depicted as areas of deep blue or green hues, reflective surfaces, and possibly showing variations like waves or currents."
        ]

        label_map = np.squeeze(
            sio.loadmat('/media/xd132/USER_new/HQG/gqh/ArbilitraryTune_continue/data_augsburg/All_Label.mat')[
                'All_Label'].astype(np.float32))

    # Initialize environment
    set_seed(SEED)
    # Prepare dataset and model
    train_dataset = LIDARHS(patchsize=patchsize_inthis, mode='train')
    train_loader = DataLoader(
        train_dataset,
        batch_size=TRAIN_PARAMS['batch_size'],
        shuffle=True,
        num_workers=1,
        drop_last=True
    )
    # Set up training components
    optimizer = optim.Adam(model.parameters(), lr=TRAIN_PARAMS['learning_rate'])
    criterion = nn.CrossEntropyLoss()
    # Start training
    train_model(model, train_loader, criterion, optimizer)