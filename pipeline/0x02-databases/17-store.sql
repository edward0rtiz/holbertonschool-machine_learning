-- query that decreases the quantity of an item after adding a new order
delimiter $$
CREATE TRIGGER decrease_items
    AFTER INSERT
    ON orders
    FOR EACH ROW
        BEGIN
            UPDATE items
            SET items.quantity = items.quantity - NEW.number
            WHERE items.name = NEW.item_name;

        END;
$$
delimiter ;