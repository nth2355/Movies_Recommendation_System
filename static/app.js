const input = document.getElementById("searchInput");
const suggestBox = document.getElementById("suggestBox");
const result = document.getElementById("result");

let selected = null;

//AUTOCOMPLETE
input.addEventListener("input", async () => {
  const q = input.value;

  if (!q) {
    suggestBox.innerHTML = "";
    return;
  }

  const res = await fetch(`/suggest?q=${q}`);
  const data = await res.json();

  if (data.length === 0) {
    suggestBox.innerHTML =
      "<div class='suggest-item'>Không tìm thấy phim</div>";
    return;
  }

  suggestBox.innerHTML = data
    .map(
      (item) =>
        `<div class="suggest-item" onclick="selectMovie('${item.replace(/'/g, "\\'")}')">
            ${item}
        </div>`,
    )
    .join("");
});

//SELECT MOVIE
function selectMovie(title) {
  input.value = title;
  suggestBox.innerHTML = "";
  selected = title;
}

//ENTER SEARCH
input.addEventListener("keydown", async (e) => {
  if (e.key === "Enter") {
    const title = input.value;

    const res = await fetch("/search", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ title }),
    });

    const data = await res.json();

    if (data.error === "not_found") {
      result.innerHTML = `<p>Không tìm thấy phim bạn vừa nhập</p>`;
      return;
    }

    result.innerHTML = data
      .map(
        (movie) => `
            <div class="movie-card">
                <h3>${movie.title}</h3>
                <p>⭐ ${movie.vote_average} | 📅 ${movie.release_date}</p>
            </div>
        `,
      )
      .join("");
  }
});
