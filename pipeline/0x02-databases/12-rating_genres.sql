-- query to list all genres by their rating
SELECT tv_genres.name AS name, SUM(rate) AS rating
FROM tv_genres
JOIN tv_show_genres
    ON tv_show_genres.genre_id = tv_genres.id
JOIN tv_show_ratings
    ON tv_show_ratings.show_id = tv_show_genres.show_id
GROUP BY name
ORDER BY rating DESC;
