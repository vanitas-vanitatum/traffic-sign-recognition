import src.data_preprocessing.rule_engine as r

FLIP_H = r.Flip('horizontal')
FLIP_V = r.Flip('vertical')
ROT_90 = r.Rotate('90')
ROT_180 = r.Rotate('180')
ROT_270 = r.Rotate('270')
TRA_BACK = r.Transpose('\\')
TRA_FRONT = r.Transpose('/')


FULLY_SYMMETRIC_INFERENCES = [FLIP_H, FLIP_V, ROT_90, ROT_180, ROT_270, TRA_BACK, TRA_FRONT]
VERTICAL_SYMMETRIC_INFERENCES = [FLIP_V, ROT_180]
HORIZONTAL_SYMMETRIC_INFERENCES = [FLIP_H, ROT_180]
VERT_HOR_SYMMETRIC_INFERENCES = [FLIP_H, FLIP_V, ROT_180]


def spawn_rules_common_predicate(predicate, inferences):
    return [predicate.then(inference) for inference in inferences]


CLASSES = ['bend_left', 'bend_right', 'children', 'closed_both_directions',
           'give_way', 'hazard', 'left_only', 'no_entry', 'priority_road',
           'right_only', 'roadworks', 'roundabout', 'stop', 'straight_ahead_only',
           'straight_or_left_only', 'straight_or_right_only', 'traffic_lights',
           'keep_right', 'keep_left', 'no_overtaking', 'wildlife', 'children',
           'slippery_road', 'no_parking', 'no_stopping', 'speed_bump', 'one_way',
           'dead_end', 'end_priority_road', 'pedestrian_crossing']

SPEED_LIMIT_CLASSES = ['speed_limit', 'speed_limit_5',
                       'speed_limit_15', 'speed_limit_20',
                       'speed_limit_30', 'speed_limit_40',
                       'speed_limit_50', 'speed_limit_60',
                       'speed_limit_80', 'speed_limit_90',
                       'speed_limit_100', 'speed_limit_120',
                       'end_of_speed_limit_50',
                       'end_of_speed_limit_40',
                       'end_of_speed_limit']

CHINESE_TO_EXCLUDE = ['bend_left', 'bend_right', 'stop']
CLASSES_TO_INCLUDE_FROM_ALL = list(set(CLASSES) - set(CHINESE_TO_EXCLUDE))


RULESET = [r.IfClassIn(CLASSES_TO_INCLUDE_FROM_ALL).then(r.Identity()),
           (r.IfDatasetIsnt('TSRD') & r.IfClassIn(CHINESE_TO_EXCLUDE)).then(r.Identity()),

           r.IfClassStartsWith('speed_limit').then(r.ChangeLabel('speed_limit')),
           r.IfClassStartsWith('end_of_speed_limit').then(r.ChangeLabel('end_of_speed_limit')),

           r.IfClassIs('straight_or_left_only').then(FLIP_H + r.ChangeLabel('straight_or_right_only')),
           r.IfClassIs('straight_or_right_only').then(FLIP_H + r.ChangeLabel('straight_or_left_only')),

           (r.IfDatasetIsnt('TSRD') & r.IfClassIs('bend_left')).then(FLIP_H + r.ChangeLabel('bend_right')),
           (r.IfDatasetIsnt('TSRD') & r.IfClassIs('bend_right')).then(FLIP_H + r.ChangeLabel('bend_left')),

           r.IfClassIs('right_only').then(FLIP_H + r.ChangeLabel('left_only')),
           (r.IfDatasetIs('Belgium') & r.IfClassIs('right_only')).then(ROT_180 + r.ChangeLabel('left_only')),
           (r.IfDatasetIs('Belgium') & r.IfClassIs('right_only')).then(ROT_270 + r.ChangeLabel('straight_ahead_only')),
           (r.IfDatasetIs('Belgium') & r.IfClassIs('right_only')).then(TRA_FRONT + r.ChangeLabel('straight_ahead_only')),
           (r.IfDatasetIs('Belgium') & r.IfClassIs('right_only')).then(FLIP_V),

           r.IfClassIs('left_only').then(FLIP_H + r.ChangeLabel('right_only')),
           (r.IfDatasetIs('Belgium') & r.IfClassIs('left_only')).then(ROT_180 + r.ChangeLabel('right_only')),
           (r.IfDatasetIs('Belgium') & r.IfClassIs('left_only')).then(ROT_90 + r.ChangeLabel('straight_ahead_only')),
           (r.IfDatasetIs('Belgium') & r.IfClassIs('left_only')).then(TRA_BACK + r.ChangeLabel('straight_ahead_only')),
           (r.IfDatasetIs('Belgium') & r.IfClassIs('left_only')).then(FLIP_V),

           r.IfClassIs('straight_ahead_only').then(ROT_90 + r.ChangeLabel('right_only')),
           r.IfClassIs('straight_ahead_only').then(ROT_270 + r.ChangeLabel('left_only')),
           r.IfClassIs('straight_ahead_only').then(TRA_FRONT + r.ChangeLabel('right_only')),
           r.IfClassIs('straight_ahead_only').then(TRA_BACK + r.ChangeLabel('left_only')),
           r.IfClassIs('straight_ahead_only').then(FLIP_H),

           r.IfClassIs('traffic_lights').then(FLIP_H),
           r.IfClassIs('give_way').then(FLIP_H),
           r.IfClassIs('hazard').then(FLIP_H),

           r.IfClassIs('keep_right').then(FLIP_H + r.ChangeLabel('keep_left')),
           r.IfClassIs('keep_right').then(ROT_90 + r.ChangeLabel('keep_left')),
           r.IfClassIs('keep_left').then(FLIP_H + r.ChangeLabel('keep_right')),
           r.IfClassIs('keep_left').then(ROT_270 + r.ChangeLabel('keep_right')),

           r.IfClassIs('dead_end').then(FLIP_H),
           r.IfClassIs('speed_bump').then(FLIP_H),
           r.IfClassIs('slippery_road').then(FLIP_H),
           r.IfClassIs('one_way').then(FLIP_H),
          
           r.IfClassIs('warning_pedestrians').then(r.ChangeLabel('pedestrian_crossing'))]


RULESET = (
        RULESET
        + spawn_rules_common_predicate(r.IfClassIs('no_stopping'), FULLY_SYMMETRIC_INFERENCES)
        + spawn_rules_common_predicate(r.IfClassIs('no_parking'), [TRA_BACK, TRA_FRONT, ROT_180])
        + spawn_rules_common_predicate(r.IfClassIs('end_priority_road'), [TRA_BACK, TRA_FRONT, ROT_180])
        + spawn_rules_common_predicate(r.IfClassIs('priority_road'), FULLY_SYMMETRIC_INFERENCES)
        + spawn_rules_common_predicate(r.IfClassIs('closed_both_directions'), FULLY_SYMMETRIC_INFERENCES)
        + spawn_rules_common_predicate(r.IfClassIs('no_entry'), VERT_HOR_SYMMETRIC_INFERENCES)
        + spawn_rules_common_predicate(r.IfClassIs('roundabout'), FULLY_SYMMETRIC_INFERENCES)
)