WITH pairs AS (
 SELECT user_id,item_id,
 MIN(CASE WHEN behavior_type=1 THEN event_time END) pv,
 MIN(CASE WHEN behavior_type=2 THEN event_time END) fav,
 MIN(CASE WHEN behavior_type=3 THEN event_time END) cart,
 MIN(CASE WHEN behavior_type=4 THEN event_time END) first_buy,
 MAX(CASE WHEN behavior_type=4 THEN event_time END) last_buy
 FROM user_behavior INDEXED BY idx_behavior_quadruple
 GROUP BY user_id,item_id
) SELECT COUNT(*) AS all_interaction_pairs,
SUM(first_buy IS NOT NULL) AS purchased_pairs,
SUM(pv<first_buy) AS pv_before_first_buy,
SUM(cart<first_buy) AS cart_before_first_buy,
SUM(fav<first_buy) AS fav_before_first_buy,
SUM(pv=first_buy) AS earliest_pv_same_hour_first_buy,
SUM(cart=first_buy) AS earliest_cart_same_hour_first_buy,
SUM(fav=first_buy) AS earliest_fav_same_hour_first_buy,
SUM(first_buy IS NOT NULL AND cart<first_buy
     AND (fav IS NULL OR fav>=first_buy)) AS cart_only_before_first_buy,
SUM(first_buy IS NOT NULL AND fav<first_buy
     AND (cart IS NULL OR cart>=first_buy)) AS fav_only_before_first_buy,
SUM(cart<first_buy AND fav<first_buy) AS both_before_first_buy,
SUM(first_buy IS NOT NULL AND (cart IS NULL OR cart>=first_buy)
     AND (fav IS NULL OR fav>=first_buy)) AS neither_earlier_before_first_buy,
SUM(first_buy IS NOT NULL AND (cart IS NULL OR cart>first_buy)
     AND (fav IS NULL OR fav>first_buy) AND pv<first_buy)
     AS observed_browse_without_cart_fav_before_or_same_hour,
SUM(first_buy IS NOT NULL AND (cart IS NULL OR cart>=first_buy)
     AND (fav IS NULL OR fav>=first_buy)
     AND (cart=first_buy OR fav=first_buy)) AS no_earlier_intent_but_same_hour_intent,
SUM(pv IS NOT NULL) AS pv_pairs,
SUM(pv<last_buy) AS pv_later_buy_pairs,
SUM(pv<=:cutoff) AS pv_complete7d_pairs,
SUM(CASE WHEN pv<=:cutoff AND last_buy>pv THEN
        EXISTS(SELECT 1 FROM user_behavior b INDEXED BY idx_behavior_quadruple
        WHERE b.user_id=pairs.user_id AND b.item_id=pairs.item_id
          AND b.behavior_type=4 AND b.event_time>pairs.pv
          AND b.event_time<=datetime(pairs.pv,'+7 days'))
        ELSE 0 END) AS pv_complete7d_buy_pairs,
SUM(cart IS NOT NULL) AS cart_pairs,
SUM(cart<last_buy) AS cart_later_buy_pairs,
SUM(cart<=:cutoff) AS cart_complete7d_pairs,
SUM(CASE WHEN cart<=:cutoff AND last_buy>cart THEN
        EXISTS(SELECT 1 FROM user_behavior b INDEXED BY idx_behavior_quadruple
        WHERE b.user_id=pairs.user_id AND b.item_id=pairs.item_id
          AND b.behavior_type=4 AND b.event_time>pairs.cart
          AND b.event_time<=datetime(pairs.cart,'+7 days'))
        ELSE 0 END) AS cart_complete7d_buy_pairs,
SUM(fav IS NOT NULL) AS fav_pairs,
SUM(fav<last_buy) AS fav_later_buy_pairs,
SUM(fav<=:cutoff) AS fav_complete7d_pairs,
SUM(CASE WHEN fav<=:cutoff AND last_buy>fav THEN
        EXISTS(SELECT 1 FROM user_behavior b INDEXED BY idx_behavior_quadruple
        WHERE b.user_id=pairs.user_id AND b.item_id=pairs.item_id
          AND b.behavior_type=4 AND b.event_time>pairs.fav
          AND b.event_time<=datetime(pairs.fav,'+7 days'))
        ELSE 0 END) AS fav_complete7d_buy_pairs FROM pairs