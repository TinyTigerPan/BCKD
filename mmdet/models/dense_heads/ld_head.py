# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmcv.runner import force_fp32

from mmdet.core import bbox_overlaps, multi_apply, reduce_mean, anchor_inside_flags, unmap, images_to_levels
from mmdet.core.bbox.iou_calculators import build_iou_calculator
from ..builder import HEADS, build_loss
from .gfl_head import GFLHead


@HEADS.register_module()
class LDHead(GFLHead):
    """Localization distillation Head. (Short description)

    It utilizes the learned bbox distributions to transfer the localization
    dark knowledge from teacher to student. Original paper: `Localization
    Distillation for Object Detection. <https://arxiv.org/abs/2102.12252>`_

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        loss_ld (dict): Config of Localization Distillation Loss (LD),
            T is the temperature for distillation.
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 loss_ld=dict(
                     type='KnowledgeDistillationKLDivLoss',
                     loss_weight=0.25,
                     T=10),
                 loss_ld_vlr=dict(
                     type='KnowledgeDistillationKLDivLoss',
                     loss_weight=0.25,
                     T=10),
                 loss_kd=dict(
                     type='KnowledgeDistillationKLDivLoss',
                     loss_weight=10,
                     T=2),
                 loss_kd_vlr=dict(
                     type='KnowledgeDistillationKLDivLoss',
                     loss_weight=10,
                     T=2),
                 loss_score_kd=dict(
                     type='MSELoss',
                     loss_weight=0.),
                 loss_score_kd_vlr=dict(
                     type='MSELoss',
                     loss_weight=0.),
                 loss_im=dict(type='IMLoss', loss_weight=0),
                 imitation_method='gibox',
                 **kwargs):

        super(LDHead, self).__init__(num_classes, in_channels, **kwargs)
        self.imitation_method = imitation_method
        self.loss_im = build_loss(loss_im)
        self.loss_ld = build_loss(loss_ld)
        self.loss_ld_vlr = build_loss(loss_ld_vlr)
        self.loss_kd = build_loss(loss_kd)
        self.loss_kd_vlr = build_loss(loss_kd_vlr)
        self.loss_score_kd = build_loss(loss_score_kd)
        self.loss_score_kd_vlr = build_loss(loss_score_kd_vlr)
        self.iou_calculator = build_iou_calculator(
            dict(type='BboxOverlaps2D'), )

    def loss_single(self, anchors, cls_score, bbox_pred, labels, label_weights,
                    bbox_targets, stride, soft_targets, soft_label, x,
                    teacher_x, vlr_region, im_region, num_total_samples):
        """Compute loss of a single scale level.

        Args:
            anchors (Tensor): Box reference for each scale level with shape
                (N, num_total_anchors, 4).
            cls_score (Tensor): Cls and quality joint scores for each scale
                level has shape (N, num_classes, H, W).
            bbox_pred (Tensor): Box distribution logits for each scale
                level with shape (N, 4*(n+1), H, W), n is max value of integral
                set.
            labels (Tensor): Labels of each anchors with shape
                (N, num_total_anchors).
            label_weights (Tensor): Label weights of each anchor with shape
                (N, num_total_anchors)
            bbox_targets (Tensor): BBox regression targets of each anchor
                weight shape (N, num_total_anchors, 4).
            stride (tuple): Stride in this scale level.
            num_total_samples (int): Number of positive samples that is
                reduced over all GPUs.

        Returns:
            dict[tuple, Tensor]: Loss components and weight targets.
        """
        assert stride[0] == stride[1], 'h stride is not equal to w stride!'
        anchors = anchors.reshape(-1, 4)
        cls_score = cls_score.permute(0, 2, 3,
                                      1).reshape(-1, self.cls_out_channels)
        bbox_pred = bbox_pred.permute(0, 2, 3,
                                      1).reshape(-1, 4 * (self.reg_max + 1))
        soft_targets = soft_targets.permute(0, 2, 3,
                                            1).reshape(-1,
                                                       4 * (self.reg_max + 1))
        soft_label = soft_label.permute(0, 2, 3,
                                        1).reshape(-1, self.cls_out_channels)
        teacher_x = teacher_x.permute(0, 2, 3, 1).reshape(-1, 256)
        x = x.permute(0, 2, 3, 1).reshape(-1, 256)

        bbox_targets = bbox_targets.reshape(-1, 4)
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        vlr_region = vlr_region.reshape(-1)
        im_region = im_region.reshape(-1)

        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        bg_class_ind = self.num_classes
        pos_inds = ((labels >= 0)
                    & (labels < bg_class_ind)).nonzero().squeeze(1)
        score = label_weights.new_zeros(labels.shape)
        score_t = label_weights.new_zeros(labels.shape)
        remain_inds = (vlr_region > 0).nonzero().squeeze(1)
        
        if self.imitation_method == 'gibox':
            gi_idx = self.get_gi_region(soft_label, cls_score, anchors,
                                        bbox_pred, soft_targets, stride)
            gi_teacher = teacher_x[gi_idx]
            gi_student = x[gi_idx]

            loss_im = self.loss_im(gi_student, gi_teacher)
        elif self.imitation_method == 'decouple':
            fg_inds = (im_region > 0).nonzero().squeeze(1)
            ng_inds = (im_region == 0).nonzero().squeeze(1)
            if len(fg_inds) > 0:
                loss_im = self.loss_im(x[fg_inds],
                                       teacher_x[fg_inds]) + 2 * self.loss_im(
                                           x[ng_inds], teacher_x[fg_inds])
            else:
                loss_im = bbox_pred.sum() * 0
        else:
            fg_inds = (im_region > 0).nonzero().squeeze(1)
            if len(fg_inds) > 0:
                loss_im = self.loss_im(x[fg_inds], teacher_x[fg_inds])
            else:
                loss_im = bbox_pred.sum() * 0

        if len(pos_inds) > 0:
            pos_bbox_targets = bbox_targets[pos_inds]
            pos_bbox_pred = bbox_pred[pos_inds]
            pos_anchors = anchors[pos_inds]
            pos_anchor_centers = self.anchor_center(pos_anchors) / stride[0]

            weight_targets = cls_score.detach().sigmoid() + 1e-4
            weight_targets = weight_targets.max(dim=1)[0][pos_inds] + 1e-4
            pos_bbox_pred_corners = self.integral(pos_bbox_pred)
            pos_decode_bbox_pred = self.bbox_coder.decode(
                pos_anchor_centers, pos_bbox_pred_corners)
            pos_decode_bbox_targets = pos_bbox_targets / stride[0]
            score[pos_inds] = bbox_overlaps(
                pos_decode_bbox_pred.detach(),
                pos_decode_bbox_targets,
                is_aligned=True)
            
            anchor_centers = self.anchor_center(anchors) / (stride[0])
            total_pred_corners = self.integral(bbox_pred)
            total_soft_corners = self.integral(soft_targets)
            total_decode_bbox_pred = self.bbox_coder.decode(anchor_centers, total_pred_corners)
            total_decode_bbox_pred_t = self.bbox_coder.decode(anchor_centers, total_soft_corners)
            weight_targets1 = cls_score.detach().sigmoid()
            weight_targets2 = soft_label.detach().sigmoid()
            weight_targets3 = weight_targets2 - weight_targets1
            total_weight_targets = weight_targets3.abs().pow(1.0).max(dim=1)[0] * label_weights

            loss_score_kd = self.loss_score_kd(
                total_decode_bbox_pred,
                total_decode_bbox_pred_t.detach(),
                weight=total_weight_targets,
                avg_factor=num_total_samples)

            loss_score_kd_vlr = 0. * self.loss_score_kd_vlr(
                total_decode_bbox_pred[remain_inds],
                total_decode_bbox_pred_t[remain_inds].detach(),
                weight=total_weight_targets[remain_inds],
                avg_factor=1.0)
            
            pred_corners = pos_bbox_pred.reshape(-1, self.reg_max + 1)
            pos_soft_targets = soft_targets[pos_inds]
            soft_corners = pos_soft_targets.reshape(-1, self.reg_max + 1)

            target_corners = self.bbox_coder.encode(pos_anchor_centers,
                                                    pos_decode_bbox_targets,
                                                    self.reg_max).reshape(-1)

            # regression loss
            loss_bbox = self.loss_bbox(
                pos_decode_bbox_pred,
                pos_decode_bbox_targets,
                weight=weight_targets,
                avg_factor=1.0)

            # dfl loss
            loss_dfl = self.loss_dfl(
                pred_corners,
                target_corners,
                weight=weight_targets[:, None].expand(-1, 4).reshape(-1),
                avg_factor=4.0)

            # ld loss
            loss_ld = self.loss_ld(
                pred_corners,
                soft_corners,
                weight=weight_targets[:, None].expand(-1, 4).reshape(-1),
                avg_factor=4.0)
            
            loss_kd = self.loss_kd(
                cls_score,
                soft_label,
                weight=label_weights,
                avg_factor=num_total_samples,)

        else:
            loss_ld = bbox_pred.sum() * 0
            loss_bbox = bbox_pred.sum() * 0
            loss_dfl = bbox_pred.sum() * 0
            loss_kd = bbox_pred.sum() * 0
            loss_im = bbox_pred.sum() * 0
            loss_score_kd = bbox_pred.sum() * 0
            loss_score_kd_vlr = bbox_pred.sum() * 0
            weight_targets = bbox_pred.new_tensor(0)

        if len(remain_inds) > 0:
            neg_pred_corners = bbox_pred[remain_inds].reshape(
                -1, self.reg_max + 1)
            neg_soft_corners = soft_targets[remain_inds].reshape(
                -1, self.reg_max + 1)

            remain_targets = vlr_region[remain_inds]

            loss_ld_vlr = self.loss_ld_vlr(
                neg_pred_corners,
                neg_soft_corners,
                weight=remain_targets[:, None].expand(-1, 4).reshape(-1),
                avg_factor=16.0)
            loss_kd_neg = bbox_pred.sum() * 0
        else:
            loss_ld_vlr = bbox_pred.sum() * 0
            loss_kd_neg = bbox_pred.sum() * 0
            
        # cls (qfl) loss
        loss_cls = self.loss_cls(
            cls_score, (labels, score),
            weight=label_weights,
            avg_factor=num_total_samples)

        return loss_cls, loss_bbox, loss_dfl, loss_ld, loss_ld_vlr, loss_kd, loss_kd_neg, loss_im, loss_score_kd, loss_score_kd_vlr, weight_targets.sum()

    def forward_train(self,
                      x,
                      out_teacher,
                      teacher_x,
                      img_metas,
                      gt_bboxes,
                      gt_labels=None,
                      gt_bboxes_ignore=None,
                      proposal_cfg=None,
                      **kwargs):
        """
        Args:
            x (list[Tensor]): Features from FPN.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes (Tensor): Ground truth bboxes of the image,
                shape (num_gts, 4).
            gt_labels (Tensor): Ground truth labels of each box,
                shape (num_gts,).
            gt_bboxes_ignore (Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).
            proposal_cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used

        Returns:
            tuple[dict, list]: The loss components and proposals of each image.

            - losses (dict[str, Tensor]): A dictionary of loss components.
            - proposal_list (list[Tensor]): Proposals of each image.
        """
        outs = self(x)
        if gt_labels is None:
            loss_inputs = outs + (gt_bboxes, out_teacher, x, img_metas)
        else:
            loss_inputs = outs + (gt_bboxes, gt_labels, out_teacher, x,
                                  teacher_x, img_metas)
        losses = self.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        if proposal_cfg is None:
            return losses
        else:
            proposal_list = self.get_bboxes(*outs, img_metas, cfg=proposal_cfg)
            return losses, proposal_list

    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def loss(self,
             cls_scores,
             bbox_preds,
             gt_bboxes,
             gt_labels,
             soft_teacher,
             x,
             teacher_x,
             img_metas,
             gt_bboxes_ignore=None):
        """Compute losses of the head.

        Args:
            cls_scores (list[Tensor]): Cls and quality scores for each scale
                level has shape (N, num_classes, H, W).
            bbox_preds (list[Tensor]): Box distribution logits for each scale
                level with shape (N, 4*(n+1), H, W), n is max value of integral
                set.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (list[Tensor] | None): specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """

        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.prior_generator.num_levels

        device = cls_scores[0].device
        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, img_metas, device=device)
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        soft_label, soft_target = soft_teacher

        cls_reg_targets = self.get_targets(
            anchor_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=label_channels)
        if cls_reg_targets is None:
            return None

        (anchor_list, labels_list, label_weights_list, bbox_targets_list,
         bbox_weights_list, num_total_pos, num_total_neg, assigned_neg_list,
         im_region_list) = cls_reg_targets

        num_total_samples = reduce_mean(
            torch.tensor(num_total_pos, dtype=torch.float,
                         device=device)).item()
        num_total_samples = max(num_total_samples, 1.0)

        losses_cls, losses_bbox, losses_dfl, losses_ld, losses_ld_vlr, losses_kd, losses_kd_neg, losses_im, \
        losses_score_kd, loss_score_kd_vlr, avg_factor = multi_apply(
            self.loss_single,
            anchor_list,
            cls_scores,
            bbox_preds,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            self.anchor_generator.strides,
            soft_target,
            soft_label,
            x,
            teacher_x,
            assigned_neg_list,
            im_region_list,
            num_total_samples=num_total_samples,
        )

        avg_factor = sum(avg_factor) + 1e-6
        avg_factor = reduce_mean(avg_factor).item()
        losses_bbox = [x / avg_factor for x in losses_bbox]
        losses_dfl = [x / avg_factor for x in losses_dfl]
        return dict(
            loss_cls=losses_cls,
            loss_bbox=losses_bbox,
            loss_dfl=losses_dfl,
            loss_ld=losses_ld,
            loss_ld_vlr=losses_ld_vlr,
            loss_kd=losses_kd,
            loss_kd_neg=losses_kd_neg,
            loss_im=losses_im,
            loss_score_kd=losses_score_kd,
            loss_score_kd_vlr=loss_score_kd_vlr,
        )
        
    def get_targets(self,
                    anchor_list,
                    valid_flag_list,
                    gt_bboxes_list,
                    img_metas,
                    gt_bboxes_ignore_list=None,
                    gt_labels_list=None,
                    label_channels=1,
                    unmap_outputs=True):
        """Get targets for GFL head.

        This method is almost the same as `AnchorHead.get_targets()`. Besides
        returning the targets as the parent method does, it also returns the
        anchors as the first element of the returned tuple.
        """
        num_imgs = len(img_metas)
        assert len(anchor_list) == len(valid_flag_list) == num_imgs

        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        num_level_anchors_list = [num_level_anchors] * num_imgs

        # concat all level anchors and flags to a single tensor
        for i in range(num_imgs):
            assert len(anchor_list[i]) == len(valid_flag_list[i])
            anchor_list[i] = torch.cat(anchor_list[i])
            valid_flag_list[i] = torch.cat(valid_flag_list[i])

        # compute targets for each image
        if gt_bboxes_ignore_list is None:
            gt_bboxes_ignore_list = [None for _ in range(num_imgs)]
        if gt_labels_list is None:
            gt_labels_list = [None for _ in range(num_imgs)]

        (all_anchors, all_labels, all_label_weights, all_bbox_targets,
         all_bbox_weights, pos_inds_list, neg_inds_list, all_vlr_region,
         all_im_region) = multi_apply(
            self._get_target_single,
            anchor_list,
            valid_flag_list,
            num_level_anchors_list,
            gt_bboxes_list,
            gt_bboxes_ignore_list,
            gt_labels_list,
            img_metas,
            label_channels=label_channels,
            unmap_outputs=unmap_outputs)

        # no valid anchors
        if any([labels is None for labels in all_labels]):
            return None
        # sampled anchors of all images
        num_total_pos = sum([max(inds.numel(), 1) for inds in pos_inds_list])
        num_total_neg = sum([max(inds.numel(), 1) for inds in neg_inds_list])
        #num_total_remain_neg = sum([max(inds.numel(), 1) for inds in assigned_neg_inds_list])
        # split targets to a list w.r.t. multiple levels
        anchors_list = images_to_levels(all_anchors, num_level_anchors)
        labels_list = images_to_levels(all_labels, num_level_anchors)
        label_weights_list = images_to_levels(all_label_weights,
                                              num_level_anchors)
        bbox_targets_list = images_to_levels(all_bbox_targets,
                                             num_level_anchors)
        bbox_weights_list = images_to_levels(all_bbox_weights,
                                             num_level_anchors)
        vlr_regions_list = images_to_levels(all_vlr_region, num_level_anchors)
        im_regions_list = images_to_levels(all_im_region, num_level_anchors)
        # sampled anchors of all images

        return (anchors_list, labels_list, label_weights_list,
                bbox_targets_list, bbox_weights_list, num_total_pos,
                num_total_neg, vlr_regions_list, im_regions_list)

    def _get_target_single(self,
                           flat_anchors,
                           valid_flags,
                           num_level_anchors,
                           gt_bboxes,
                           gt_bboxes_ignore,
                           gt_labels,
                           img_meta,
                           label_channels=1,
                           unmap_outputs=True):
        """Compute regression, classification targets for anchors in a single
        image.

        Args:
            flat_anchors (Tensor): Multi-level anchors of the image, which are
                concatenated into a single tensor of shape (num_anchors, 4)
            valid_flags (Tensor): Multi level valid flags of the image,
                which are concatenated into a single tensor of
                    shape (num_anchors,).
            num_level_anchors Tensor): Number of anchors of each scale level.
            gt_bboxes (Tensor): Ground truth bboxes of the image,
                shape (num_gts, 4).
            gt_bboxes_ignore (Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).
            gt_labels (Tensor): Ground truth labels of each box,
                shape (num_gts,).
            img_meta (dict): Meta info of the image.
            label_channels (int): Channel of label.
            unmap_outputs (bool): Whether to map outputs back to the original
                set of anchors.

        Returns:
            tuple: N is the number of total anchors in the image.
                anchors (Tensor): All anchors in the image with shape (N, 4).
                labels (Tensor): Labels of all anchors in the image with shape
                    (N,).
                label_weights (Tensor): Label weights of all anchor in the
                    image with shape (N,).
                bbox_targets (Tensor): BBox targets of all anchors in the
                    image with shape (N, 4).
                bbox_weights (Tensor): BBox weights of all anchors in the
                    image with shape (N, 4).
                pos_inds (Tensor): Indices of postive anchor with shape
                    (num_pos,).
                neg_inds (Tensor): Indices of negative anchor with shape
                    (num_neg,).
        """
        inside_flags = anchor_inside_flags(flat_anchors, valid_flags,
                                           img_meta['img_shape'][:2],
                                           self.train_cfg.allowed_border)
        if not inside_flags.any():
            return (None, ) * 7
        # assign gt and sample anchors
        anchors = flat_anchors[inside_flags, :]

        num_level_anchors_inside = self.get_num_level_anchors_inside(
            num_level_anchors, inside_flags)

        assign_result = self.assigner.assign(anchors, num_level_anchors_inside,
                                             gt_bboxes, gt_bboxes_ignore,
                                             gt_labels)

        sampling_result = self.sampler.sample(assign_result, anchors,
                                              gt_bboxes)

        vlr_region = self.assigner.get_vlr_region(anchors,
                                                  num_level_anchors_inside,
                                                  gt_bboxes, gt_bboxes_ignore,
                                                  gt_labels)

        im_region = self.get_im_region(
            anchors, gt_bboxes, mode=self.imitation_method)

        num_valid_anchors = anchors.shape[0]
        bbox_targets = torch.zeros_like(anchors)
        bbox_weights = torch.zeros_like(anchors)

        labels = anchors.new_full((num_valid_anchors, ),
                                  self.num_classes,
                                  dtype=torch.long)
        labels_neg = anchors.new_full((num_valid_anchors, ),
                                      self.num_classes,
                                      dtype=torch.long)

        label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)

        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        if len(pos_inds) > 0:
            pos_bbox_targets = sampling_result.pos_gt_bboxes
            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1.0

            if gt_labels is None:
                # Only rpn gives gt_labels as None
                # Foreground is the first class
                labels[pos_inds] = 0
            else:
                labels[pos_inds] = gt_labels[
                    sampling_result.pos_assigned_gt_inds]
            if self.train_cfg.pos_weight <= 0:
                label_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = self.train_cfg.pos_weight
        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        # map up to original set of anchors
        if unmap_outputs:
            num_total_anchors = flat_anchors.size(0)
            anchors = unmap(anchors, num_total_anchors, inside_flags)
            labels = unmap(
                labels, num_total_anchors, inside_flags, fill=self.num_classes)
            label_weights = unmap(label_weights, num_total_anchors,
                                  inside_flags)
            bbox_targets = unmap(bbox_targets, num_total_anchors, inside_flags)
            bbox_weights = unmap(bbox_weights, num_total_anchors, inside_flags)
            vlr_region = unmap(vlr_region, num_total_anchors, inside_flags)
            im_region = unmap(im_region, num_total_anchors, inside_flags)

            labels_neg = unmap(
                labels_neg,
                num_total_anchors,
                inside_flags,
                fill=self.num_classes)

        return (anchors, labels, label_weights, bbox_targets, bbox_weights,
                pos_inds, neg_inds, vlr_region, im_region)
        
    # imitation region
    def get_im_region(self, bboxes, gt_bboxes, mode='fitnet'):
        assert mode in ['gibox', 'finegrained', 'fitnet', 'decouple']
        num_gt, num_bboxes = gt_bboxes.size(0), bboxes.size(0)

        # compute iou between all bbox and gt

        overlaps = self.iou_calculator(bboxes, gt_bboxes)
        bboxes = bboxes[:, :4]
        assigned_gt_inds = overlaps.new_full((num_bboxes, ),
                                             0,
                                             dtype=torch.long)
        assigned_fg = (assigned_gt_inds + 0).float()
        # compute iou between all bbox and gt

        iou = self.iou_calculator(bboxes, gt_bboxes, mode='iou')
        fine_grained = torch.nonzero(iou > 0.5 * iou.max(0)[0])
        assigned_fg[fine_grained[:, 0]] = 1
        gt_flag = torch.zeros(bboxes.shape[0])
        anchor_center = self.anchor_center(bboxes)
        for gt_bbox in gt_bboxes:
            in_gt_flag = torch.nonzero(
                (anchor_center[:, 0] > gt_bbox[0])
                & (anchor_center[:, 0] < gt_bbox[2])
                & (anchor_center[:, 1] > gt_bbox[1])
                & (anchor_center[:, 1] < gt_bbox[3]),
                as_tuple=False)
            gt_flag[in_gt_flag] = 1

        if mode == 'finegrained':
            return assigned_fg
        else:
            return gt_flag

    def get_gi_region(self, soft_label, cls_score, anchors, bbox_pred,
                      soft_targets, stride):

        teacher_score = soft_label.detach().sigmoid()

        student_score = cls_score.detach().sigmoid()  #[num,80]

        anchor_centers = self.anchor_center(anchors) / stride[0]
        sdistribution = self.integral(bbox_pred)
        tdistribution = self.integral(soft_targets)
        sbox = self.bbox_coder.decode(anchor_centers, sdistribution)  #[num,4]
        tbox = self.bbox_coder.decode(anchor_centers, tdistribution)

        z = teacher_score - student_score  #difference between teacher score and student score on the whole locations.
        giscore, index = torch.abs(z).max(dim=1)  #GI scores
        k = z >= 0  #who is bigger
        j = torch.take(
            k, index + self.cls_out_channels *
            (torch.arange(student_score.size(0)).cuda()))
        h = j == 0
        gibox = sbox.new_zeros(sbox.shape)
        gibox[j] = tbox[j] + 0
        gibox[h] = sbox[h] + 0  #GI boxes

        idx_out = torch.ops.torchvision.nms(gibox, giscore, 0.3)[:10]
        return idx_out
