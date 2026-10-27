"""对保留正常重复的 SQLite 执行描述性分析。"""

QUERIES: dict[str, str] = {
    "behavior_distribution": """
        SELECT behavior_type, COUNT(*) event_count,
               COUNT(DISTINCT user_id) users,
               1.0*COUNT(*)/SUM(COUNT(*)) OVER() event_share
        FROM user_behavior GROUP BY behavior_type
    """,
    "daily_activity": """
        WITH user_days AS MATERIALIZED (
          SELECT DISTINCT user_id,date(event_time) day FROM user_behavior
        ), days AS (SELECT DISTINCT date(event_time) day FROM dim_time)
        SELECT day,
          (SELECT COUNT(*) FROM user_days u WHERE u.day=days.day) dau,
          (SELECT COUNT(DISTINCT user_id) FROM user_days u
           WHERE u.day>=date(days.day,'-6 days') AND u.day<=days.day) rolling_7d_wau,
          day>=(SELECT date(MIN(event_time),'+6 days') FROM dim_time)
            complete_7d_window
        FROM days ORDER BY day
    """,
    "hour_activity": """
        SELECT CAST(strftime('%H',event_time) AS INTEGER) hour,
          COUNT(*) event_count, COUNT(DISTINCT user_id) active_users,
          COUNT(DISTINCT CASE WHEN behavior_type=4 THEN user_id END) buyers,
          1.0*COUNT(DISTINCT CASE WHEN behavior_type=4 THEN user_id END)
            /COUNT(DISTINCT user_id) buyer_user_share
        FROM user_behavior GROUP BY hour ORDER BY hour
    """,
    "weekday_weekend": """
        WITH daily AS (
          SELECT date(event_time) day,
            CAST(strftime('%w',event_time) IN ('0','6') AS INTEGER) is_weekend,
            COUNT(*) events, COUNT(DISTINCT user_id) users,
            COUNT(DISTINCT CASE WHEN behavior_type=4 THEN user_id END) buyers,
            SUM(behavior_type=4) buys
          FROM user_behavior GROUP BY day
        )
        SELECT is_weekend, COUNT(*) observed_days, AVG(events) events_per_day,
          AVG(users) dau, AVG(buys) buys_per_day,
          AVG(1.0*buyers/users) mean_daily_buyer_share
        FROM daily GROUP BY is_weekend
    """,
    "user_features": """
        WITH daily AS (
          SELECT user_id,date(event_time) day, COUNT(*) events
          FROM user_behavior GROUP BY user_id,day
        ), islands AS (
          SELECT user_id,day,julianday(day)-ROW_NUMBER() OVER(
            PARTITION BY user_id ORDER BY day) island FROM daily
        ), streaks AS (
          SELECT user_id,island,COUNT(*) streak_days FROM islands
          GROUP BY user_id,island
        ), longest AS (
          SELECT user_id,MAX(streak_days) longest_active_streak FROM streaks
          GROUP BY user_id
        ), aggregate_users AS (
          SELECT user_id,COUNT(*) events,COUNT(DISTINCT item_id) distinct_items,
            COUNT(DISTINCT date(event_time)) active_days,
            SUM(behavior_type=1) pv_count,SUM(behavior_type=2) fav_count,
            SUM(behavior_type=3) cart_count,SUM(behavior_type=4) buy_count,
            MIN(event_time) first_time,MAX(event_time) last_time,
            MAX(CASE WHEN behavior_type=4 THEN event_time END) last_buy_time,
            SUM(CASE behavior_type WHEN 4 THEN 5 WHEN 3 THEN 3
                  WHEN 2 THEN 2 ELSE 1 END) preference_weight_sum
          FROM user_behavior GROUP BY user_id
        )
        SELECT a.*,l.longest_active_streak,
          julianday((SELECT datetime(MAX(event_time),'+1 hour') FROM dim_time))
            -julianday(last_time) inactivity_days,
          julianday((SELECT datetime(MAX(event_time),'+1 hour') FROM dim_time))
            -julianday(last_buy_time) purchase_recency_days
        FROM aggregate_users a JOIN longest l USING(user_id)
    """,
    "ordered_funnel": """
        WITH pv AS (
          SELECT user_id,item_id,MIN(event_time) t1 FROM user_behavior
          WHERE behavior_type=1 GROUP BY user_id,item_id
        ), cart AS (
          SELECT p.user_id,p.item_id,p.t1,MIN(b.event_time) t2 FROM pv p
          JOIN user_behavior b ON b.user_id=p.user_id AND b.item_id=p.item_id
            AND b.behavior_type=3 AND b.event_time>p.t1
          GROUP BY p.user_id,p.item_id
        ), fav AS (
          SELECT c.user_id,c.item_id,c.t1,c.t2,MIN(b.event_time) t3 FROM cart c
          JOIN user_behavior b ON b.user_id=c.user_id AND b.item_id=c.item_id
            AND b.behavior_type=2 AND b.event_time>c.t2
          GROUP BY c.user_id,c.item_id
        ), buy AS (
          SELECT f.user_id,f.item_id,MIN(b.event_time) t4 FROM fav f
          JOIN user_behavior b ON b.user_id=f.user_id AND b.item_id=f.item_id
            AND b.behavior_type=4 AND b.event_time>f.t3
          GROUP BY f.user_id,f.item_id
        )
        SELECT 'pv' stage,COUNT(*) user_item_pairs FROM pv
        UNION ALL SELECT 'pv_cart',COUNT(*) FROM cart
        UNION ALL SELECT 'pv_cart_fav',COUNT(*) FROM fav
        UNION ALL SELECT 'pv_cart_fav_buy',COUNT(*) FROM buy
    """,
    "category_analysis": """
        WITH pair AS (
          SELECT b.user_id,b.item_id,i.category_id,
            SUM(behavior_type=1) pv,SUM(behavior_type=4) buys,
            MIN(CASE WHEN behavior_type=1 THEN event_time END) first_pv,
            MAX(CASE WHEN behavior_type=4 THEN event_time END) last_buy,
            COUNT(DISTINCT CASE WHEN behavior_type=4 THEN date(event_time) END)
              purchase_days
          FROM user_behavior b JOIN dim_item i USING(item_id)
          GROUP BY b.user_id,b.item_id
        ), ordered AS (
          SELECT p.*,(SELECT MIN(b.event_time) FROM user_behavior b
            WHERE b.user_id=p.user_id AND b.item_id=p.item_id
              AND b.behavior_type=4 AND b.event_time>p.first_pv) first_later_buy
          FROM pair p
        )
        SELECT category_id,COUNT(DISTINCT item_id) items,COUNT(*) interacting_pairs,
          SUM(pv) pv_count,SUM(buys) buy_events,SUM(buys>0) buyer_pairs,
          SUM(pv>0) viewed_pairs,SUM(first_later_buy IS NOT NULL) converted_pairs,
          1.0*SUM(first_later_buy IS NOT NULL)/NULLIF(SUM(pv>0),0)
            strict_later_hour_pair_conversion,
          1.0*SUM(purchase_days>=2)/NULLIF(SUM(buys>0),0)
            cross_day_repeat_buyer_pair_share,
          AVG((julianday(first_later_buy)-julianday(first_pv))*24)
            first_pv_to_first_later_buy_hours
        FROM ordered GROUP BY category_id ORDER BY buy_events DESC
    """,
    "item_lifecycle": """
        WITH bounds AS (SELECT datetime(MAX(event_time),'+1 hour') finish
                        FROM dim_time), counts AS (
          SELECT item_id,COUNT(*) events,COUNT(DISTINCT user_id) users,
            SUM(behavior_type=4) buys,
            SUM(event_time>=datetime(finish,'-3 days')) recent_3d_events,
            SUM(event_time>=datetime(finish,'-7 days') AND
                event_time<datetime(finish,'-3 days')) previous_4d_events,
            MAX(event_time) last_time
          FROM user_behavior CROSS JOIN bounds GROUP BY item_id
        ), cat AS (
          SELECT i.category_id,COUNT(*) observed_category_items,
            SUM(c.events) category_events,
            SUM(1.0*c.events*c.events) squared_events
          FROM counts c JOIN dim_item i USING(item_id) GROUP BY i.category_id
        )
        SELECT c.*,i.category_id,cat.observed_category_items,
          1.0*c.events/cat.category_events category_event_share,
          cat.squared_events/(1.0*cat.category_events*cat.category_events)
            category_concentration_hhi,
          ((recent_3d_events+1.0)/3)/((previous_4d_events+1.0)/4)
            recent_heat_ratio,
          (julianday((SELECT finish FROM bounds))-julianday(last_time))*24
            item_inactivity_hours
        FROM counts c JOIN dim_item i USING(item_id) JOIN cat USING(category_id)
        ORDER BY buys DESC
    """,
    "hour_sequence": """
        WITH bins AS (
          SELECT user_id,event_time,
            MAX(behavior_type=1)+2*MAX(behavior_type=2)
              +4*MAX(behavior_type=3)+8*MAX(behavior_type=4) mask,
            COUNT(*) bin_events
          FROM user_behavior GROUP BY user_id,event_time
        ), sequence AS (
          SELECT *,LAG(event_time) OVER w previous_time,LAG(mask) OVER w previous_mask
          FROM bins WINDOW w AS(PARTITION BY user_id ORDER BY event_time)
        )
        SELECT user_id,event_time,mask,bin_events,previous_mask,
          (julianday(event_time)-julianday(previous_time))*24 gap_hours
        FROM sequence
    """,
}
