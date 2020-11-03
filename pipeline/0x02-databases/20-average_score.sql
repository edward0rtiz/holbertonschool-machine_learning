-- procedure to store the average score of a student
DELIMITER $$
CREATE PROCEDURE ComputeAverageScoreForUser(
    IN user_id_new INT)
BEGIN
    UPDATE users SET average_score=(
        SELECT AVG(score)
        FROM corrections
        WHERE user_id = user_id_new)
        WHERE id=user_id_new;
END $$
DELIMITER ;