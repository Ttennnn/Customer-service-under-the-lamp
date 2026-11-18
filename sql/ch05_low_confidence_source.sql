SET NAMES utf8mb4;
ALTER TABLE low_confidence_questions
  MODIFY source ENUM('retrieval_low_conf','self_check','user_feedback','confidence_gate')
  NOT NULL COMMENT '入池入口:检索证据低 / 生成自评不足 / 用户反馈未解决 / ch05 置信度闸知识缺口';
