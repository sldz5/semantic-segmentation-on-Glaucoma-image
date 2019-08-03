import numpy as np
#from prepare1 import _encode_segmap
import scipy.misc as m
def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) +
        label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)
    return hist

def accuracy_score(label_trues, label_preds, n_class=3):
    """Returns accuracy score evaluation result.

      - overall accuracy
      - mean accuracy
      - mean IU
      - fwavacc
    """
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    mean_iu = np.nanmean(iu)
    return  mean_iu  

def newiou(pred,target):
    ans = []
    for cls in range(1,3):
        pred_inds = pred >= cls
        target_inds = target >= cls
        intersection = pred_inds[target_inds].sum()
        union = pred_inds.sum() + target_inds.sum() - intersection
        result=float(intersection) / max(union, 1)
        ans.append(result)
    return np.array(ans)

def _get_pascal_labels():
        return np.asarray([[0,0,0], [255,0,0], [0,255,0]])

def _encode_segmap(npgt):
    # npgt contains 255
    npgt = npgt.astype(int)

    npgt2 = np.zeros((npgt.shape[0], npgt.shape[1]), dtype=np.int16)
    for i, label in enumerate(_get_pascal_labels()):
        npgt2[np.where(np.all(npgt == label, axis=-1))[:2]] = i
    npgt2 = npgt2.astype(int)
    return npgt2

if __name__ == '__main__':
    mious=0
    quaniou = np.array([0.0, 0.0])
    for i in range(1, 11):
        target = m.imread('./256fcn/testB/'+('%d' % i)+'.png')
        target = _encode_segmap(target)
        #########################################################
        predict = m.imread('./output/'+('%d' % i)+'.png')
        predict = _encode_segmap(predict)
        a = newiou(predict, target)
        quaniou+=a
        b=accuracy_score(predict,target)
        mious+=b
        print("{}-IOU:{},disc:{},cup:{}".format(i,b,a[0],a[1]))
    print(mious/10)
    print(quaniou/10)