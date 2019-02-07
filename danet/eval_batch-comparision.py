def eval_batch(image, dst, evaluator, eval_mode):
    if eval_mode:
        # evaluation mode on validation set
        targets = dst
        outputs = evaluator.parallel_forward(image)

        batch_inter, batch_union, batch_correct, batch_label = 0, 0, 0, 0
        for output, target in zip(outputs, targets):
            correct, labeled = utils.batch_pix_accuracy(output.data.cpu(), target)
            inter, union = utils.batch_intersection_union(
                output.data.cpu(), target, testset.num_class)
            batch_correct += correct
            batch_label += labeled
            batch_inter += inter
            batch_union += union
        return batch_correct, batch_label, batch_inter, batch_union
    else:
        # Visualize and dump the results
        im_paths = dst
        outputs = evaluator.parallel_forward(image)
        predicts = [torch.max(output, 1)[1].cpu().numpy() + testset.pred_offset
                    for output in outputs]
        for predict, impath in zip(predicts, im_paths):
            mask = utils.get_mask_pallete(predict, args.dataset)
            outname = os.path.splitext(impath)[0] + '.png'
            mask.save(os.path.join(outdir, outname))
        # dummy outputs for compatible with eval mode
        return 0, 0, 0, 0


def eval_batch(model, image, target):
    outputs = model(image)
    outputs = gather(outputs, 0, dim=0)
    pred = outputs[0]
    target = target.cuda()
    correct, labeled = utils.batch_pix_accuracy(pred.data, target)
    inter, union = utils.batch_intersection_union(pred.data, target, self.nclass)
    return correct, labeled, inter, union