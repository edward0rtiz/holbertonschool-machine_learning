-- query to reset attribute valid_email
delimiter $$
CREATE TRIGGER reset_email
    BEFORE UPDATE
    ON users
    FOR EACH ROW
        BEGIN
            IF OLD.email <> NEW.email THEN
                SET NEW.valid_email = 0;
            END IF;
        END;
$$
delimiter;
