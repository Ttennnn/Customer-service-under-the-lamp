ALTER TABLE tickets
  MODIFY COLUMN ticket_type ENUM('售后','投诉','咨询','退款') NOT NULL;
