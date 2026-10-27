-- 新版清洗保留正常重复，四元组只建普通索引；event_id 仍为唯一主键。
CREATE INDEX idx_behavior_quadruple
ON user_behavior(user_id, item_id, behavior_type, event_time);
CREATE INDEX idx_behavior_user_time ON user_behavior(user_id, event_time);
CREATE INDEX idx_behavior_item_type_user ON user_behavior(item_id, behavior_type, user_id);
CREATE INDEX idx_behavior_type_time ON user_behavior(behavior_type, event_time);
CREATE INDEX idx_behavior_time_user ON user_behavior(event_time, user_id);
CREATE INDEX idx_item_category ON dim_item(category_id);
ANALYZE;
