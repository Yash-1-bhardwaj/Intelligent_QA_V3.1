class Fusion:

    def combine(self, text_results, image_results):

        scores = {}

        # Text results
        for rank, page_id in enumerate(
            text_results["ids"][0],
            start=1
        ):
            scores.setdefault(page_id, 0)

            scores[page_id] += 1 / rank

        # Image results
        for rank, page_id in enumerate(
            image_results["ids"][0],
            start=1
        ):
            scores.setdefault(page_id, 0)

            scores[page_id] += 1 / rank

        # Sort highest score first
        ranked_pages = sorted(
            scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return ranked_pages